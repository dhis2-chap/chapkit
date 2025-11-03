"""Vega-Lite visualization service demo.

This example demonstrates:
- Custom Router for data transformation (DataFrame → Vega-Lite spec)
- Multiple chart types (line, bar, scatter, heatmap, aggregations)
- Data processing (filtering, aggregation, pivoting)
- RESTful API design with $ prefix for operations
- Proof of concept for Phase 5.1 (Transformation Service Pattern)
"""

from typing import Any

import altair as alt  # type: ignore[import-not-found]
import pandas as pd
import vl_convert as vlc  # type: ignore[import-not-found]
from fastapi import HTTPException, Response, status
from pydantic import BaseModel, Field
from servicekit.api import BaseServiceBuilder, ServiceInfo
from servicekit.api.router import Router

from chapkit.data import DataFrame


class VegaGenerateRequest(BaseModel):
    """Request to generate a Vega-Lite specification from data."""

    data: DataFrame
    chart_type: str = Field(
        ...,
        description="Chart type: line, bar, scatter, heatmap, boxplot, histogram",
    )
    x_field: str | None = Field(None, description="Field for x-axis")
    y_field: str | None = Field(None, description="Field for y-axis")
    color_field: str | None = Field(None, description="Field for color encoding")
    title: str | None = Field(None, description="Chart title")
    width: int = Field(600, description="Chart width in pixels")
    height: int = Field(400, description="Chart height in pixels")
    aggregate: str | None = Field(
        None,
        description="Aggregation function: mean, sum, count, median, min, max",
    )
    format: str = Field("json", description="Output format: json, png, svg, html")


class VegaAggregateRequest(BaseModel):
    """Request to aggregate data and generate visualization."""

    data: DataFrame
    group_by: list[str] = Field(..., description="Fields to group by")
    agg_field: str = Field(..., description="Field to aggregate")
    agg_func: str = Field(..., description="Aggregation function: mean, sum, count, etc.")
    chart_type: str = Field("bar", description="Chart type for aggregated data")
    title: str | None = None
    format: str = Field("json", description="Output format: json, png, svg, html")


class VegaResponse(BaseModel):
    """Response containing Vega-Lite specification."""

    spec: dict[str, Any] = Field(..., description="Vega-Lite specification")
    row_count: int = Field(..., description="Number of data rows")
    columns: list[str] = Field(..., description="Data columns")


class VegaRouter(Router):
    """Router for generating Vega-Lite visualizations from DataFrame."""

    def _register_routes(self) -> None:
        """Register Vega visualization endpoints."""

        @self.router.post(
            "/$generate",
            response_model=None,
            status_code=status.HTTP_200_OK,
            summary="Generate Vega-Lite spec or rendered image",
            description="Transform DataFrame to Vega-Lite specification or rendered format (json/png/svg/html)",
        )
        async def generate_vega(request: VegaGenerateRequest) -> VegaResponse | Response:
            """Generate Vega-Lite specification or rendered image from data."""
            # Convert to pandas DataFrame
            df = request.data.to_pandas()

            # Validate fields exist
            if request.x_field and request.x_field not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Field '{request.x_field}' not found in data",
                )
            if request.y_field and request.y_field not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Field '{request.y_field}' not found in data",
                )

            # Build chart
            chart = self._build_chart(
                df=df,
                chart_type=request.chart_type,
                x_field=request.x_field,
                y_field=request.y_field,
                color_field=request.color_field,
                title=request.title,
                width=request.width,
                height=request.height,
                aggregate=request.aggregate,
            )

            # Return based on format
            if request.format == "json":
                spec = chart.to_dict()
                return VegaResponse(
                    spec=spec,
                    row_count=len(df),
                    columns=df.columns.tolist(),
                )
            else:
                return self._render_chart(chart, request.format)

        @self.router.post(
            "/$aggregate",
            response_model=None,
            status_code=status.HTTP_200_OK,
            summary="Aggregate and visualize",
            description="Aggregate DataFrame and generate visualization in specified format",
        )
        async def aggregate_and_visualize(request: VegaAggregateRequest) -> VegaResponse | Response:
            """Aggregate data and generate visualization."""
            # Convert to pandas DataFrame
            df = request.data.to_pandas()

            # Validate fields
            for field in request.group_by:
                if field not in df.columns:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Group field '{field}' not found in data",
                    )
            if request.agg_field not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Aggregation field '{request.agg_field}' not found in data",
                )

            # Perform aggregation
            agg_df = self._aggregate_data(
                df=df,
                group_by=request.group_by,
                agg_field=request.agg_field,
                agg_func=request.agg_func,
            )

            # Generate chart for aggregated data
            x_field = request.group_by[0] if request.group_by else None
            y_field = f"{request.agg_field}_{request.agg_func}"

            chart = self._build_chart(
                df=agg_df,
                chart_type=request.chart_type,
                x_field=x_field,
                y_field=y_field,
                color_field=request.group_by[1] if len(request.group_by) > 1 else None,
                title=request.title or f"{request.agg_func.title()} of {request.agg_field}",
                width=600,
                height=400,
            )

            # Return based on format
            if request.format == "json":
                spec = chart.to_dict()
                return VegaResponse(
                    spec=spec,
                    row_count=len(agg_df),
                    columns=agg_df.columns.tolist(),
                )
            else:
                return self._render_chart(chart, request.format)

    def _build_chart(
        self,
        df: pd.DataFrame,
        chart_type: str,
        x_field: str | None,
        y_field: str | None,
        color_field: str | None,
        title: str | None,
        width: int,
        height: int,
        aggregate: str | None = None,
    ) -> alt.Chart:
        """Build altair chart object."""
        # Create base chart with data
        chart = alt.Chart(df).properties(width=width, height=height)

        if title:
            chart = chart.properties(title=title)

        # Build encoding based on chart type
        if chart_type == "line":
            chart = chart.mark_line(point=True, tooltip=True)
            chart = self._add_encoding(chart, x_field, y_field, color_field, aggregate)

        elif chart_type == "bar":
            chart = chart.mark_bar(tooltip=True)
            chart = self._add_encoding(chart, x_field, y_field, color_field, aggregate)

        elif chart_type == "scatter":
            chart = chart.mark_point(tooltip=True)
            chart = self._add_encoding(chart, x_field, y_field, color_field, aggregate)

        elif chart_type == "heatmap":
            chart = chart.mark_rect()
            if x_field and y_field:
                encoding = {"x": alt.X(x_field, type="nominal"), "y": alt.Y(y_field, type="nominal")}
                if aggregate and y_field:
                    encoding["color"] = alt.Color(y_field, type="quantitative", aggregate=aggregate)
                elif color_field:
                    encoding["color"] = alt.Color(color_field, type="quantitative")
                chart = chart.encode(**encoding)

        elif chart_type == "boxplot":
            chart = chart.mark_boxplot(extent="min-max")
            chart = self._add_encoding(chart, x_field, y_field, color_field, aggregate=None)

        elif chart_type == "histogram":
            field = x_field or y_field
            if field:
                chart = chart.mark_bar().encode(
                    x=alt.X(field, type="quantitative", bin=True), y=alt.Y("count()", type="quantitative")
                )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported chart type: {chart_type}",
            )

        return chart

    def _add_encoding(
        self,
        chart: alt.Chart,
        x_field: str | None,
        y_field: str | None,
        color_field: str | None,
        aggregate: str | None,
    ) -> alt.Chart:
        """Add encoding to altair chart."""
        encoding: dict[str, Any] = {}

        if x_field:
            encoding["x"] = alt.X(x_field, type="nominal")

        if y_field:
            if aggregate:
                encoding["y"] = alt.Y(y_field, type="quantitative", aggregate=aggregate)
            else:
                encoding["y"] = alt.Y(y_field, type="quantitative")

        if color_field:
            encoding["color"] = alt.Color(color_field, type="nominal")

        return chart.encode(**encoding)

    def _render_chart(self, chart: alt.Chart, format: str) -> Response:
        """Render chart to specified format."""
        if format == "png":
            png_data = chart.to_json()
            png_bytes = vlc.vegalite_to_png(png_data, scale=2)
            return Response(content=png_bytes, media_type="image/png")

        elif format == "svg":
            svg_data = chart.to_json()
            svg_str = vlc.vegalite_to_svg(svg_data)
            return Response(content=svg_str, media_type="image/svg+xml")

        elif format == "html":
            html_str = chart.to_html()
            return Response(content=html_str, media_type="text/html")

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}. Use json, png, svg, or html",
            )

    def _aggregate_data(
        self,
        df: pd.DataFrame,
        group_by: list[str],
        agg_field: str,
        agg_func: str,
    ) -> pd.DataFrame:
        """Aggregate DataFrame by specified fields."""
        if agg_func == "count":
            agg_df = df.groupby(group_by).size().reset_index(name=f"{agg_field}_count")  # pyright: ignore[reportUnknownMemberType]
        else:
            agg_mapping = {f"{agg_field}_{agg_func}": (agg_field, agg_func)}
            agg_df = df.groupby(group_by).agg(**agg_mapping).reset_index()  # pyright: ignore[reportUnknownMemberType]

        return agg_df


# Create router instance
vega_router = VegaRouter.create(
    prefix="/api/v1/visualizations",
    tags=["visualizations"],
)

# Build FastAPI application
app = (
    BaseServiceBuilder(
        info=ServiceInfo(
            display_name="Vega Visualization Service",
            version="1.0.0",
            summary="Transform DataFrame to Vega-Lite specifications",
            description="Proof of concept for Phase 5.1 Transformation Service Pattern. "
            "Accepts DataFrame input and generates Vega-Lite grammar for various chart types. "
            "Supports data processing including aggregation, filtering, and transformations.",
        )
    )
    .with_health()
    .with_system()
    .with_logging()
    .with_monitoring()
    .with_landing_page()
    .include_router(vega_router)
    .build()
)


if __name__ == "__main__":
    from servicekit.api.utilities import run_app

    run_app(app, reload=False)
