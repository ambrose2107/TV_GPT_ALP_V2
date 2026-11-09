import tempfile
import unittest
from pathlib import Path
import pandas as pd
from scripts.xauusd_data_pipeline import read_file, normalize_frame, resample_ohlc, write_monthly, validate_library, import_files

class PipelineTests(unittest.TestCase):
    def test_date_time_utc(self):
        raw=pd.DataFrame({"Date":["2025-01-02","2025-01-02"],"Time":["00:00:00","00:01:00"],
          "Open":[2650,2651],"High":[2652,2653],"Low":[2649,2650],"Close":[2651,2652],"Volume":[10,12]})
        out=normalize_frame(raw,"forexsb","UTC")
        self.assertEqual(len(out),2); self.assertEqual(str(out.index.tz),"UTC")
    def test_histdata_headerless_ascii_m1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"DAT_ASCII_XAUUSD_M1_202401.csv"
            path.write_text(
                "20240102 000000;2060.10;2060.40;2059.90;2060.20;0\\n"
                "20240102 000100;2060.20;2060.50;2060.00;2060.30;0\\n",
                encoding="utf-8")
            raw=read_file(path)
            out=normalize_frame(raw,"histdata","Etc/GMT+5")
            self.assertEqual(len(out),2)
            self.assertAlmostEqual(out.iloc[0]["open"],2060.10)
            self.assertEqual(str(out.index.tz),"UTC")
            self.assertEqual(out.index[0].hour,5)

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
    def test_bid_ask_imports_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"sample.csv"
            pd.DataFrame({"time":["2025-01-01T00:00:00Z"],"open":[1],"high":[2],"low":[.5],"close":[1.5]}).to_csv(source,index=False)
            import_files([source],"dukascopy",root/"XAUUSD",None,"M1","BID")
            pd.DataFrame({"time":["2025-01-01T00:00:00Z"],"open":[1.1],"high":[2.1],"low":[.6],"close":[1.6]}).to_csv(source,index=False)
            import_files([source],"dukascopy",root/"XAUUSD",None,"M1","ASK")
            bid=root/"XAUUSD"/"BID"/"normalized"/"M1"/"2025"/"XAUUSD_M1_2025_01.csv.gz"
            ask=root/"XAUUSD"/"ASK"/"normalized"/"M1"/"2025"/"XAUUSD_M1_2025_01.csv.gz"
            self.assertTrue(bid.exists()); self.assertTrue(ask.exists())
            self.assertNotEqual(pd.read_csv(bid,compression="gzip").iloc[0]["open"],
                                pd.read_csv(ask,compression="gzip").iloc[0]["open"])
    def test_monthly_validate(self):
        idx=pd.date_range("2025-01-31 23:58",periods=5,freq="min",tz="UTC")
        df=pd.DataFrame({"open":[1]*5,"high":[2]*5,"low":[.5]*5,"close":[1.5]*5,"volume":[10]*5},index=idx)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); write_monthly(df,"M1",root)
            report=validate_library(root); self.assertEqual(report["status"],"PASS")
            self.assertEqual(report["timeframes"][0]["bars"],5)

if __name__=="__main__": unittest.main()


class DukascopyDecoderTests(unittest.TestCase):
    def test_decode_bi5_m1_record(self):
        import lzma
        import struct
        from datetime import date
        from scripts.download_dukascopy_xauusd import decode_bi5

        record = struct.pack(">5If", 0, 2060100, 2060200, 2060000, 2060150, 3.5)
        payload = lzma.compress(record, format=lzma.FORMAT_ALONE)
        out = decode_bi5(payload, date(2025, 1, 2))
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["time"], pd.Timestamp("2025-01-02T00:00:00Z"))
        self.assertAlmostEqual(out.iloc[0]["open"], 2060.1)
        self.assertAlmostEqual(out.iloc[0]["high"], 2060.2)
        self.assertAlmostEqual(out.iloc[0]["low"], 2060.0)
        self.assertAlmostEqual(out.iloc[0]["close"], 2060.15)
        self.assertAlmostEqual(out.iloc[0]["volume"], 3.5)

    def test_decode_rejects_truncated_record(self):
        import lzma
        from datetime import date
        from scripts.download_dukascopy_xauusd import decode_bi5

        with self.assertRaisesRegex(ValueError, "Malformed BI5"):
            decode_bi5(lzma.compress(b"bad", format=lzma.FORMAT_ALONE), date(2025, 1, 2))
