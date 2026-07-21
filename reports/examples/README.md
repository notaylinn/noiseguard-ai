# Example Reports

`example_report.pdf` in this folder shows the exact layout produced by
`backend/app/services/report_service.py` (`PdfReportService`), rendered
here from representative sample data so reviewers can see report
formatting without first running the full ML pipeline.

Once the app is running, every real analysis gets its own PDF at:

```
GET /api/v1/reports/{analysis_id}/pdf
```

or via the "Download PDF Report" button in the Streamlit UI.
