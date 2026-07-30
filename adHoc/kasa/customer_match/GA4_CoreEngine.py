import polars as pl
import numpy as np
from math import ceil
from typing import List, Optional, Literal
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (DateRange, Dimension, Metric, FilterExpression,
                                                Filter, RunReportRequest, NumericValue, OrderBy)

class BuildReport:
    def __init__(self, property_id: str, ga_dimensions: List[str], ga_metrics: List[str],
                 start_date: str, end_date: str, creds_path: Optional[str] = None) -> None:
        """
        This builds a GA4 report that can be run with or without a filter

        Dimension and metrics can be found by visiting:
        https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema

        start_date
            The inclusive start date for the query in the format
            ``YYYY-MM-DD``. Cannot be after ``end_date``. The format
            ``NdaysAgo``, ``yesterday``, or ``today`` is also accepted,
            and in that case, the date is inferred based on the
            property's reporting time zone.

        end_date
            The inclusive end date for the query in the format
            ``YYYY-MM-DD``. Cannot be before ``start_date``. The format
            ``NdaysAgo``, ``yesterday``, or ``today`` is also accepted,
            and in that case, the date is inferred based on the
            property's reporting time zone.

        SAMPLE CODE

        from GoogleAnalytics4 import GA4

        report = GA4.BuildReport(property_id='123456789',
                         ga_dimensions=['pagePath', 'pageTitle'],
                         ga_metrics=['screenPageViews', 'activeUsers', 'averageSessionDuration'],
                         start_date='2023-02-01',
                         end_date='today')

        :param property_id: GA4 property id
        :param ga_dimensions: list of GA4 dimensions you want to return
        :param ga_metrics: list of GA4 metrics you want to return
        :param start_date: pull data starting from this date
        :param end_date: pull data ending on this date
        :param creds_path: if specified use ga4-gtm-automation.json path and not the environment variable
        """
        self.dimension_filter = None
        self.metric_filter    = None
        self.dimensions       = [Dimension(name=x) for x in ga_dimensions]
        self.metrics          = [Metric(name=x) for x in ga_metrics]
        self.date_ranges      = [DateRange(start_date=start_date, end_date=end_date)]
        self.property_id      = property_id
        
        if creds_path:
            self.client = BetaAnalyticsDataClient.from_service_account_file(creds_path)
        else:   
            self.client = BetaAnalyticsDataClient()

    def add_filter(self,
                   filter_type: Literal['string_filter', 'in_list_filter', 'numeric_filter', 'between_filter'],
                   filter_dimension: bool,
                   field_name: str,
                   filter_values: Optional[List[str] | str | NumericValue] = None,
                   filter_case: Optional[bool] = False,
                   match_type: Optional[Filter.StringFilter.MatchType] = Filter.StringFilter.MatchType(0),
                   operation: Optional[Filter.NumericFilter.Operation] = Filter.NumericFilter.Operation(0),
                   from_value: Optional[NumericValue] = None,
                   to_value: Optional[NumericValue] = None) -> None:
        """
        This adds a filter to the BuildReport object. This is not required - i.e., BuildReport objects can be run
        without a filter

        The match_type of a string filter
            MATCH_TYPE_UNSPECIFIED = 0
            EXACT = 1
            BEGINS_WITH = 2
            ENDS_WITH = 3
            CONTAINS = 4
            FULL_REGEXP = 5
            PARTIAL_REGEXP = 6

        The operation applied to a numeric filter
            OPERATION_UNSPECIFIED = 0
            EQUAL = 1
            LESS_THAN = 2
            LESS_THAN_OR_EQUAL = 3
            GREATER_THAN = 4
            GREATER_THAN_OR_EQUAL = 5

        SAMPLE CODE

        from GoogleAnalytics4 import GA4

        report = GA4.BuildReport(property_id='123456789',
                         ga_dimensions=['pagePath', 'pageTitle'],
                         ga_metrics=['screenPageViews', 'activeUsers', 'averageSessionDuration'],
                         start_date='2023-02-01',
                         end_date='today')

        report.add_filter(filter_type='string_filter',
                          filter_dimension=True,  # if true use a dimension field_name else use a metric field_name
                          field_name='pagePath',
                          match_type=Filter.StringFilter.MatchType.EXACT,
                          filter_values='/Page/1',
                          filter_case=True)

        :param filter_dimension: bool if False use a metric_filter
        :param filter_type: select one of the four filter types
        :param field_name: the dimensions to filter on
        :param filter_values: the value to be used in the filter
        :param filter_case: is the filter value case-sensitive
        :param match_type: only used with a StringFilter
        :param operation: only used with a NumericFilter
        :param from_value: only used with BetweenFilter
        :param to_value: only used with BetweenFilter
        """

        literals = ['string_filter', 'in_list_filter', 'numeric_filter', 'between_filter']
        if filter_type not in literals:
            raise ValueError(f"filter_type must be 'string_filter', 'in_list_filter', 'numeric_filter' "
                             f"or 'between_filter' you entered '{filter_type}'")

        filter_obj = None
        if filter_type == 'string_filter':
            filter_obj = Filter(
                field_name=field_name,
                string_filter=Filter.StringFilter(
                    match_type=match_type,
                    value=filter_values,
                    case_sensitive=filter_case
                )
            )
        elif filter_type == 'in_list_filter':
            filter_obj = Filter(
                field_name=field_name,
                in_list_filter=Filter.InListFilter(
                    values=filter_values,
                    case_sensitive=filter_case
                )
            )
        elif filter_type == 'numeric_filter':
            filter_obj = Filter(
                field_name=field_name,
                numeric_filter=Filter.NumericFilter(
                    operation=operation,
                    value=filter_values
                )
            )
        elif filter_type == 'between_filter':
            filter_obj = Filter(
                field_name=field_name,
                between_filter=Filter.BetweenFilter(
                    from_value=from_value,
                    to_value=to_value
                )
            )

        filter_expr = FilterExpression(filter=filter_obj)
        if filter_dimension:
            self.dimension_filter = filter_expr
        else:
            self.metric_filter = filter_expr
    def to_numeric(s: pl.Series) -> pl.Series:
        try:
            result = s.cast(pl.Int64)
        except pl.exceptions.InvalidOperationError:
            result = s.cast(pl.Float64)
        return result
    pl.Series.to_numeric = to_numeric

    def run_report(self, limit: int = 250000, order_bys: Optional[List[OrderBy]] = None) -> pl.DataFrame:
        """
        This is used to actually RunReportRequest, which can be used with add_filter or not

        Input value only accept python list

        SAMPLE CODE

        from GoogleAnalytics4 import GA4

        report = GA4.BuildReport(property_id='123456789',
                         ga_dimensions=['pagePath', 'pageTitle'],
                         ga_metrics=['screenPageViews', 'activeUsers', 'averageSessionDuration'],
                         start_date='2023-02-01',
                         end_date='today')

        # add_ filter is optional
        report.add_filter(filter_type='string_filter',
                          filter_dimension=True,
                          field_name='pagePath',
                          match_type=Filter.StringFilter.MatchType.EXACT,
                          filter_values='/Page/1',
                          filter_case=True)

        df = report.run_report()

        :param offset: int
        :param limit: int
        :return: Polars.DataFrame
        """
        if self.dimension_filter:
            request = RunReportRequest(property=f'properties/{self.property_id}',
                                       dimensions=self.dimensions,
                                       metrics=self.metrics,
                                       date_ranges=self.date_ranges,
                                       dimension_filter=self.dimension_filter,
                                       limit=1,
                                       order_bys=order_bys,
                                       return_property_quota= True
                                       )
        elif self.metric_filter:
            request = RunReportRequest(property=f'properties/{self.property_id}',
                                       dimensions=self.dimensions,
                                       metrics=self.metrics,
                                       date_ranges=self.date_ranges,
                                       metric_filter=self.metric_filter,
                                       limit=1,
                                       order_bys=order_bys,
                                       return_property_quota= True
                                       )
        else:
            request = RunReportRequest(property=f'properties/{self.property_id}',
                                       dimensions=self.dimensions,
                                       metrics=self.metrics,
                                       date_ranges=self.date_ranges,
                                       limit=1,
                                       order_bys=order_bys,
                                       return_property_quota= True
                                       )
            
        # added an extra minute to the timeout
        data              =  self.client.run_report(request, timeout=3600 * 2)
        total_rows        =  data.row_count
        
        # Calculate how many rows we actually want to fetch (respecting the limit)
        rows_to_fetch     =  min(total_rows, limit)
        
        # Determine batch size (can't exceed rows_to_fetch)
        # Use 100,000 as the max supported batch size for the GA4 Data API
        api_batch_size    =  min(rows_to_fetch, 100000)
        
        # Set the request limit for the batch fetch
        request.limit     =  api_batch_size
        
        # Calculate how many loops we need
        batch_loop        =  ceil(rows_to_fetch / api_batch_size) if api_batch_size > 0 else 0
        # get and declare column names for the metrics and dimensions
        dimension_headers = [header.name for header in data.dimension_headers]
        metric_headers    = [header.name for header in data.metric_headers]

        # declare a dataframe
        dfs = []
        for batch in range(batch_loop):
            df_cache = pl.DataFrame()
            request.offset = batch * api_batch_size
            
            # Ensure the last batch doesn't over-fetch if it's smaller
            if (batch + 1) == batch_loop:
                request.limit = rows_to_fetch - (batch * api_batch_size)
            
            # Auto reset
            data = self.client.run_report(request, timeout= 3600 * 2)

            # get row values
            dimension_vals = [val.value for row in data.rows for val in row.dimension_values]
            metric_vals    = [val.value for row in data.rows for val in row.metric_values]

            # assign dimension values
            df_cache[dimension_headers] = np.transpose([dimension_vals[i::len(self.dimensions)] for i in range(len(self.dimensions))])

            # assign metric values
            df_cache[metric_headers] = np.transpose([metric_vals[i::len(self.metrics)] for i in range(len(self.metrics))])

            # convert metrics to numeric to optimize memory usage
            df_cache[df_cache.columns[len(self.dimensions):]] = df_cache[df_cache.columns[len(self.dimensions):]].select( s.to_numeric() for s in df_cache[df_cache.columns[len(self.dimensions):]])
            
            # append to the list of dataframes
            dfs.append(df_cache)

        df = pl.concat(dfs, how='vertical',rechunk=True)

        return df 