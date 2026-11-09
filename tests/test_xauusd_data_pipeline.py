import tempfile
import unittest
from pathlib import Path
import pandas as pd
from scripts.xauusd_data_pipeline import normalize_frame, resample_ohlc, write_monthly, validate_library

class PipelineTests(unittest.TestCase):
    def test_date_time_utc(self):
        raw=pd.DataFrame({"Date":["2025-01-02","2025-01-02"],"Time":["00:00:00","00:01:00"],
          "Open":[2650,2651],"High":[2652,2653],"Low":[2649,2650],"Close":[2651,2652],"Volume":[10,12]})
        out=normalize_frame(raw,"forexsb","UTC")
        self.assertEqual(len(out),2); self.assertEqual(str(out.index.tz),"UTC")
    def test_naive_requires_timezone(self):
        raw=pd.DataFrame({"time":["2025-01-01 00:00:00"],"open":[1],"high":[2],"low":[.5],"close":[1.5]})
        with self.assertRaisesRegex(ValueError,"source-timezone"): normalize_frame(raw,"generic")
    def test_ohlc_invalid(self):
        raw=pd.DataFrame({"time":["2025-01-01T00:00:00Z"],"open":[2],"high":[1],"low":[.5],"close":[1.5]})
        with self.assertRaisesRegex(ValueError,"OHLC"): normalize_frame(raw,"generic")
    def test_resample(self):
        idx=pd.date_range("2025-01-01",periods=6,freq="min",tz="UTC")
        df=pd.DataFrame({"open":[1,2,3,4,5,6],"high":[2,3,4,5,6,7],"low":[.5,1.5,2.5,3.5,4.5,5.5],"close":[1.5,2.5,3.5,4.5,5.5,6.5],"volume":[1]*6},index=idx)
        out=resample_ohlc(df,"5min"); self.assertEqual(len(out),2)
        self.assertEqual(out.iloc[0]["open"],1); self.assertEqual(out.iloc[0]["close"],5.5)
        self.assertEqual(out.iloc[0]["volume"],5)
    def test_monthly_validate(self):
        idx=pd.date_range("2025-01-31 23:58",periods=5,freq="min",tz="UTC")
        df=pd.DataFrame({"open":[1]*5,"high":[2]*5,"low":[.5]*5,"close":[1.5]*5,"volume":[10]*5},index=idx)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); write_monthly(df,"M1",root)
            report=validate_library(root); self.assertEqual(report["status"],"PASS")
            self.assertEqual(report["timeframes"][0]["bars"],5)

if __name__=="__main__": unittest.main()
