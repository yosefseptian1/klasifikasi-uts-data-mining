using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using Dapper;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Npgsql;
using ScottPlot;

// === Struktur data ===
public class VitalSignData
{
    public int id { get; set; }
    public float umur { get; set; }
    public float suhu_badan { get; set; }
    public float tekanan_darah_atas { get; set; }
    public float tekanan_darah_bawah { get; set; }
    public float saturasi_oksigen { get; set; }
    public float denyut_nadi_per_menit { get; set; }
    public float respirasi_nafas_per_menit { get; set; }
    public string jenis_kelamin { get; set; } = "";
    public string kode_icd { get; set; } = "";
    public string nama_icd { get; set; } = "";
    public string kondisi { get; set; } = "";
}

public class KNNClassifier
{
    private List<(float[] Features, string Label)> _trainData = new();
    private int _k;

    public KNNClassifier(int k)
    {
        if (k <= 0) throw new ArgumentException("k harus > 0");
        _k = k;
    }

    public void Fit(IEnumerable<(float[] Features, string Label)> data)
    {
        _trainData = data.ToList();
    }

    public string Predict(float[] sample)
    {
        if (_trainData.Count == 0) throw new InvalidOperationException("Train data kosong");

        int effectiveK = Math.Min(_k, _trainData.Count);

        var neighborGroups = _trainData
            .Select(d => (Label: d.Label, Distance: EuclideanDistance(d.Features, sample)))
            .OrderBy(d => d.Distance)
            .Take(effectiveK)
            .GroupBy(d => d.Label)
            .Select(g => new { Label = g.Key, Count = g.Count(), AvgDist = g.Average(x => x.Distance) })
            .OrderByDescending(x => x.Count)      // mayoritas suara
            .ThenBy(x => x.AvgDist)              // tie-breaker: label dengan jarak rata-rata lebih kecil
            .ToList();

        return neighborGroups.First().Label;
    }

    private static float EuclideanDistance(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = a[i] - b[i];
            s += diff * diff;
        }
        return (float)Math.Sqrt(s);
    }
}

class Program
{
    // Feature extractor
    static float[] ExtractFeatures(VitalSignData d)
    {
        return new float[]
        {
            d.umur,
            d.suhu_badan,
            d.tekanan_darah_atas,
            d.tekanan_darah_bawah,
            d.denyut_nadi_per_menit,
            d.respirasi_nafas_per_menit,
            d.saturasi_oksigen
        };
    }

    // Min-Max scaler helper
    class MinMaxScaler
    {
        public float[] Min { get; private set; }
        public float[] Max { get; private set; }
        public int Dim => Min.Length;

        public MinMaxScaler() { }

        public void Fit(IEnumerable<float[]> X)
        {
            var arr = X.ToArray();
            if (arr.Length == 0) throw new ArgumentException("No data to fit scaler");
            int m = arr[0].Length;
            Min = new float[m];
            Max = new float[m];
            for (int j = 0; j < m; j++)
            {
                Min[j] = arr.Min(r => r[j]);
                Max[j] = arr.Max(r => r[j]);
            }
        }

        public float[] Transform(float[] x)
        {
            var outv = new float[Dim];
            for (int j = 0; j < Dim; j++)
            {
                if (Max[j] == Min[j]) outv[j] = 0f;
                else outv[j] = (x[j] - Min[j]) / (Max[j] - Min[j]);
            }
            return outv;
        }

        public List<float[]> Transform(IEnumerable<float[]> X)
        {
            return X.Select(x => Transform(x)).ToList();
        }
    }

    // Tentukan kondisi berdasarkan rule (note: ini membuat label rule-based)
    static string TentukanKondisi(VitalSignData d)
    {
        int abnormal = 0;

        if (d.suhu_badan < 36 || d.suhu_badan > 37.5) abnormal++;
        if (d.tekanan_darah_atas < 90 || d.tekanan_darah_atas > 120) abnormal++;
        if (d.tekanan_darah_bawah < 60 || d.tekanan_darah_bawah > 80) abnormal++;
        if (d.denyut_nadi_per_menit < 60 || d.denyut_nadi_per_menit > 100) abnormal++;
        if (d.respirasi_nafas_per_menit < 12 || d.respirasi_nafas_per_menit > 20) abnormal++;
        if (d.saturasi_oksigen < 95) abnormal++;
        
        int score = 0;

        // Respirasi
        if (d.respirasi_nafas_per_menit <= 8) score += 3;
        else if (d.respirasi_nafas_per_menit <= 11) score += 1;
        else if (d.respirasi_nafas_per_menit <= 20) score += 0;
        else if (d.respirasi_nafas_per_menit <= 24) score += 2;
        else score += 3;

        // Saturasi
        if (d.saturasi_oksigen >= 96) score += 0;
        else if (d.saturasi_oksigen >= 94) score += 1;
        else if (d.saturasi_oksigen >= 92) score += 2;
        else score += 3;

        // Heart Rate
        if (d.denyut_nadi_per_menit <= 40) score += 3;
        else if (d.denyut_nadi_per_menit <= 50) score += 1;
        else if (d.denyut_nadi_per_menit <= 90) score += 0;
        else if (d.denyut_nadi_per_menit <= 110) score += 1;
        else if (d.denyut_nadi_per_menit <= 130) score += 2;
        else score += 3;

        return abnormal switch
        {
            0 => "Normal",
            <= 2 => "Gawat",
            _ => "Darurat"
        };
    }

    // Stratified split (single split): returns (train, test)
    static (List<VitalSignData> train, List<VitalSignData> test) StratifiedSplit(List<VitalSignData> data, double trainFraction, int seed = 0)
    {
        var rnd = new Random(seed);
        var groups = data.GroupBy(d => d.kondisi);
        var train = new List<VitalSignData>();
        var test = new List<VitalSignData>();
        foreach (var g in groups)
        {
            var shuffled = g.OrderBy(_ => rnd.Next()).ToList();
            int trainCount = (int)Math.Round(shuffled.Count * trainFraction);
            if (trainCount == 0 && shuffled.Count > 0) trainCount = 1; // ensure at least one if possible
            train.AddRange(shuffled.Take(trainCount));
            test.AddRange(shuffled.Skip(trainCount));
        }
        // final shuffle
        train = train.OrderBy(_ => rnd.Next()).ToList();
        test = test.OrderBy(_ => rnd.Next()).ToList();
        return (train, test);
    }

    // Stratified k-fold: returns list of (train, test) pairs
    static List<(List<VitalSignData> train, List<VitalSignData> test)> StratifiedKFold(List<VitalSignData> data, int k, int seed = 0)
    {
        var rnd = new Random(seed);
        // Prepare folds dictionary: fold index -> list
        var folds = new List<List<VitalSignData>>();
        for (int i = 0; i < k; i++) folds.Add(new List<VitalSignData>());

        var groups = data.GroupBy(d => d.kondisi);
        foreach (var g in groups)
        {
            var shuffled = g.OrderBy(_ => rnd.Next()).ToList();
            for (int i = 0; i < shuffled.Count; i++)
            {
                folds[i % k].Add(shuffled[i]);
            }
        }

        var result = new List<(List<VitalSignData> train, List<VitalSignData> test)>();
        for (int i = 0; i < k; i++)
        {
            var test = folds[i].ToList();
            var train = folds.Where((f, idx) => idx != i).SelectMany(f => f).ToList();
            result.Add((train, test));
        }
        return result;
    }

    static void Main()
    {
        Console.WriteLine("Mengambil data vital sign dari database...");

        // ------------ 1) Koneksi DB ------------
        string connectionString = "Host=localhost; Port=5433; Database=test; Username=postgres; Password=postgres";
        List<VitalSignData> dataList;
        using (IDbConnection db = new NpgsqlConnection(connectionString))
        {
            string sql = @"SELECT * FROM pasien;";
            dataList = db.Query<VitalSignData>(sql).ToList();
        }

        // ------------- 2) Basic cleaning & clamp -------------
        dataList = dataList
            .Where(d =>
                !float.IsNaN(d.suhu_badan) &&
                !float.IsNaN(d.tekanan_darah_atas) &&
                !float.IsNaN(d.tekanan_darah_bawah) &&
                !float.IsNaN(d.denyut_nadi_per_menit) &&
                !float.IsNaN(d.respirasi_nafas_per_menit) &&
                !float.IsNaN(d.saturasi_oksigen))
            .ToList();

        foreach (var d in dataList)
        {
            d.suhu_badan = Math.Clamp(d.suhu_badan, 34, 42);
            d.tekanan_darah_atas = Math.Clamp(d.tekanan_darah_atas, 70, 200);
            d.tekanan_darah_bawah = Math.Clamp(d.tekanan_darah_bawah, 40, 130);
            d.denyut_nadi_per_menit = Math.Clamp(d.denyut_nadi_per_menit, 30, 200);
            d.respirasi_nafas_per_menit = Math.Clamp(d.respirasi_nafas_per_menit, 5, 60);
            d.saturasi_oksigen = Math.Clamp(d.saturasi_oksigen, 70, 100);
        }

        if (dataList.Count == 0)
        {
            Console.WriteLine("Tidak ada data setelah cleaning. Hentikan program.");
            return;
        }

        // -------------- 3) Buat label kondisi (rule-based) --------------
        foreach (var d in dataList)
            d.kondisi = TentukanKondisi(d);

        // -------------- 4) Shuffle data globally for ML.NET reproducibility --------------
        var rndGlobal = new Random(0);
        dataList = dataList.OrderBy(_ => rndGlobal.Next()).ToList();

        Console.WriteLine($"Total data setelah cleaning & label: {dataList.Count}");
        var total = dataList.Count;

        // -------------- 5) Prepare splits (stratified) --------------
        var split80 = StratifiedSplit(dataList, 0.8, seed: 0);
        var split90 = StratifiedSplit(dataList, 0.9, seed: 0);

        // -------------- 6) Evaluate K-NN (manual) --------------
        var resultKnn = new List<(string Model, double MicroAcc)>();
        Console.WriteLine("\n⚙ Processing model: K-NN (manual) with normalization + stratified CV");

        foreach (int k in new[] { 3, 5, 10 })
        {
            // --- 80/20 ---
            {
                var (train, test) = split80;
                // Fit scaler on train
                var scaler = new MinMaxScaler();
                var trainFeatures = train.Select(ExtractFeatures);
                scaler.Fit(trainFeatures);
                var trainTuples = train.Select(d => (scaler.Transform(ExtractFeatures(d)), d.kondisi)).ToList();
                var testTuples = test.Select(d => (scaler.Transform(ExtractFeatures(d)), d.kondisi)).ToList();

                var knn = new KNNClassifier(k);
                knn.Fit(trainTuples);

                int correct = 0;
                foreach (var (features, label) in testTuples)
                    if (knn.Predict(features) == label) correct++;

                double acc = testTuples.Count == 0 ? 0.0 : (double)correct / testTuples.Count;
                resultKnn.Add(($"KNN (k={k}, 80/20)", acc));
                Console.WriteLine($"   - KNN k={k} (80/20): {acc:P2}");
            }

            // --- 90/10 ---
            {
                var (train, test) = split90;
                var scaler = new MinMaxScaler();
                scaler.Fit(train.Select(ExtractFeatures));
                var trainTuples = train.Select(d => (scaler.Transform(ExtractFeatures(d)), d.kondisi)).ToList();
                var testTuples = test.Select(d => (scaler.Transform(ExtractFeatures(d)), d.kondisi)).ToList();

                var knn = new KNNClassifier(k);
                knn.Fit(trainTuples);

                int correct = 0;
                foreach (var (features, label) in testTuples)
                    if (knn.Predict(features) == label) correct++;

                double acc = testTuples.Count == 0 ? 0.0 : (double)correct / testTuples.Count;
                resultKnn.Add(($"KNN (k={k}, 90/10)", acc));
                Console.WriteLine($"   - KNN k={k} (90/10): {acc:P2}");
            }

            // --- Stratified K-Fold CV (3,5,10) for KNN ---
            foreach (int fold in new[] { 3, 5, 10 })
            {
                var cvPairs = StratifiedKFold(dataList, fold, seed: 0);
                double sumAcc = 0;
                int validFoldCount = 0;
                foreach (var (trainFold, testFold) in cvPairs)
                {
                    if (testFold.Count == 0) continue;

                    var scaler = new MinMaxScaler();
                    scaler.Fit(trainFold.Select(ExtractFeatures));

                    var trainTuples = trainFold.Select(d => (scaler.Transform(ExtractFeatures(d)), d.kondisi)).ToList();
                    var testTuples = testFold.Select(d => (scaler.Transform(ExtractFeatures(d)), d.kondisi)).ToList();

                    var knn = new KNNClassifier(k);
                    knn.Fit(trainTuples);

                    int correct = 0;
                    foreach (var (features, label) in testTuples)
                        if (knn.Predict(features) == label) correct++;

                    sumAcc += (double)correct / testTuples.Count;
                    validFoldCount++;
                }

                double avgAcc = validFoldCount == 0 ? 0.0 : sumAcc / validFoldCount;
                resultKnn.Add(($"KNN (k={k}, CV {fold}-fold)", avgAcc));
                Console.WriteLine($"   - KNN k={k} CV {fold}-fold avg: {avgAcc:P2}");
            }
        }

        // -------------- 7) ML.NET Models (FastTree & NaiveBayes) --------------
        // NOTE: FastTree used here is a gradient boosting tree (FastTree), not a single CART tree.
        var mlContext = new MLContext(seed: 0);
        var dataView = mlContext.Data.LoadFromEnumerable(dataList);

        var basePipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(VitalSignData.kondisi))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("jenis_kelamin"))
            .Append(mlContext.Transforms.Concatenate("Features",
                nameof(VitalSignData.umur),
                nameof(VitalSignData.suhu_badan),
                nameof(VitalSignData.tekanan_darah_atas),
                nameof(VitalSignData.tekanan_darah_bawah),
                nameof(VitalSignData.denyut_nadi_per_menit),
                nameof(VitalSignData.respirasi_nafas_per_menit),
                nameof(VitalSignData.saturasi_oksigen),
                "jenis_kelamin"))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

        // Trainers dictionary
        var models = new Dictionary<string, IEstimator<ITransformer>>
        {
            // "Decision Tree" label here uses FastTree as gradient-boosted decision trees
            { "FastTree (GradientBoosting)", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree()) },
            { "Naive Bayes", mlContext.MulticlassClassification.Trainers.NaiveBayes() },
        };

        var results = new List<(string Model, double MicroAcc, double MacroAcc, double LogLoss)>();

        foreach (var kvp in models)
        {
            Console.WriteLine($"\n⚙ Processing ML.NET model: {kvp.Key} ...");
            var pipeline = basePipeline.Append(kvp.Value)
                                       .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Split 80/20
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 0);
            var model = pipeline.Fit(split.TrainSet);
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            results.Add((kvp.Key + " (80/20)", metrics.MicroAccuracy, metrics.MacroAccuracy, metrics.LogLoss));
            Console.WriteLine($"   - {kvp.Key} (80/20) Micro: {metrics.MicroAccuracy:P2}, Macro: {metrics.MacroAccuracy:P2}");

            // Split 90/10
            var split2 = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1, seed: 0);
            var model2 = pipeline.Fit(split2.TrainSet);
            var predictions2 = model2.Transform(split2.TestSet);
            var metrics2 = mlContext.MulticlassClassification.Evaluate(predictions2);

            results.Add((kvp.Key + " (90/10)", metrics2.MicroAccuracy, metrics2.MacroAccuracy, metrics2.LogLoss));
            Console.WriteLine($"   - {kvp.Key} (90/10) Micro: {metrics2.MicroAccuracy:P2}, Macro: {metrics2.MacroAccuracy:P2}");

            // Cross Validation 3,5,10
            foreach (int fold in new[] { 3, 5, 10 })
            {
                var cvResults = mlContext.MulticlassClassification.CrossValidate(dataView, pipeline, numberOfFolds: fold, seed: 0);
                var avgMicro = cvResults.Average(r => r.Metrics.MicroAccuracy);
                var avgMacro = cvResults.Average(r => r.Metrics.MacroAccuracy);
                var avgLog = cvResults.Average(r => r.Metrics.LogLoss);

                results.Add(($"{kvp.Key} (CV {fold}-Fold)", avgMicro, avgMacro, avgLog));
                Console.WriteLine($"   - {kvp.Key} CV{fold}-fold → Micro: {avgMicro:P2}, Macro: {avgMacro:P2}");
            }
        }

        // Add KNN results to unified results (fill MacroAcc with MicroAcc for CSV parity)
        foreach (var (model, acc) in resultKnn)
            results.Add((model, acc, acc, 0));

        // -------------- 8) Save results to CSV (single write) --------------
        var csvPath = "hasil_perbandingan_model.csv";
        using (var writer = new StreamWriter(csvPath))
        {
            writer.WriteLine("Model,MicroAccuracy,MacroAccuracy,LogLoss");
            foreach (var r in results)
                writer.WriteLine($"{EscapeCsv(r.Model)},{r.MicroAcc:F6},{r.MacroAcc:F6},{r.LogLoss:F6}");
        }
        Console.WriteLine($"\n Hasil evaluasi disimpan ke: {csvPath}");

        // Tambahkan bar chart
        var plt = new ScottPlot.Plot();
        
        var modelNames = results.Select(r => r.Model).ToArray();
        var accuracies = results.Select(r => r.MicroAcc * 100).ToArray();
        double[] xs = Enumerable.Range(0, modelNames.Length).Select(i => (double)i).ToArray();
        
        var bars = plt.Add.Bars(xs, accuracies);
        plt.Axes.SetLimitsY(0, accuracies.Max() + 5);
        
        plt.Axes.Left.Label.Text = "Akurasi (%)";
        plt.Axes.Title.Label.Text = "Perbandingan Akurasi Model Klasifikasi Kondisi Pasien";
        
        plt.Axes.Bottom.SetTicks(xs, modelNames);
        plt.Axes.Bottom.TickLabelStyle.Rotation = 30;
        plt.Axes.Bottom.TickLabelStyle.Alignment = Alignment.MiddleLeft;
        
        plt.Axes.Margins(bottom: 5, top: 0.1);
        plt.Layout.Fixed(padding: new PixelPadding(left: 80, right: 40, top: 50, bottom: 110));
        
        plt.Axes.SetLimitsY(0, accuracies.Max() + 5);
        
        for (int i = 0; i < accuracies.Length; i++)
        {
            double x = xs[i];
            double y = accuracies[i] + 2;
        
            var txt = plt.Add.Text($"{accuracies[i]:0.0}%", x, y);
            txt.LabelFontSize = 14;
            txt.LabelAlignment = Alignment.UpperCenter;
        }

        // Simpan grafik sebagai PNG
        plt.SavePng("demo2.png", 1600, 800);
        Console.WriteLine("Grafik disimpan sebagai: grafik_perbandingan_model.png");
        Console.WriteLine("\n=== ✅ Analisis Selesai ===");
        
        static string EscapeCsv(string s)
        {
            if (s.Contains(',') || s.Contains('"') || s.Contains('\n')) return $"\"{s.Replace("\"", "\"\"")}\"";
            return s;
        }
    }
}