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
        _k = k;
    }
    
    public void Fit(IEnumerable<(float[] Features, string Label)> data)
    {
        _trainData = data.ToList();
    }

    public string Predict(float[] sample)
    {
        var neighbors = _trainData
            .Select(d => (Label: d.Label, Distance: EuclideanDistance(d.Features, sample)))
            .OrderBy(d => d.Distance)
            .Take(_k)
            .GroupBy(d => d.Label)
            .OrderByDescending(g => g.Count())
            .First()
            .Key;

        return neighbors;
    }

    private static float EuclideanDistance(float[] a, float[] b)
    {
        return (float)Math.Sqrt(a.Zip(b, (x, y) => Math.Pow(x - y, 2)).Sum());
    }
}


class Program
{
    static void Main()
    {
        Console.WriteLine("Mengambil data vital sign dari database...");

        // 1. Koneksi ke database PostgreSQL
        string connectionString =
            "Host=localhost; Port=5433; Database=test; Username=postgres; Password=postgres";
        using IDbConnection db = new NpgsqlConnection(connectionString);
        
        // 2. Penentuan Klasifikasi Kondisi Pasien
        static string TentukanKondisi(VitalSignData d)
        {
            int abnormal = 0;

            if (d.suhu_badan < 36 || d.suhu_badan > 37.5) abnormal++;
            if (d.tekanan_darah_atas < 90 || d.tekanan_darah_atas > 120) abnormal++;
            if (d.tekanan_darah_bawah < 60 || d.tekanan_darah_bawah > 80) abnormal++;
            if (d.denyut_nadi_per_menit < 60 || d.denyut_nadi_per_menit > 100) abnormal++;
            if (d.respirasi_nafas_per_menit < 12 || d.respirasi_nafas_per_menit > 20) abnormal++;
            if (d.saturasi_oksigen < 95) abnormal++;

            return abnormal switch
            {
                0 => "Normal",
                <= 2 => "Gawat",
                _ => "Darurat"
            };
        }

        // 3. Query Ambil data
        string sql = @"SELECT * FROM pasien;";
        var dataList = db.Query<VitalSignData>(sql).ToList();

        //4. Proses data berdasarkan jenis kondisi
        foreach (var d in dataList)
        {
            d.kondisi = TentukanKondisi(d);
        }
        
        // 5. Split data
        int total = dataList.Count;
        
        // === split data 80-20 ===//
        var split80 = (
                train: dataList.Take((int)(total * 0.8)).ToList(),
                test: dataList.Skip((int)(total * 0.8)).ToList()
            );
        
        // === split data 90-10 ===//
        var split90 = (
                train: dataList.Take((int)(total * 0.9)).ToList(),
                test: dataList.Skip((int)(total * 0.9)).ToList()
            );
        
        //6. Proses KNN Manual
        var resultKnn = new List<(string, double)>();
        Console.WriteLine($"\n⚙️ Processing data model: KNN");
        foreach (int k in new[] { 3, 5, 10 })
        {
            {
                var knn = new KNNClassifier(k);

                var trainTuples = split80.train.Select(d => (new float[]
                {
                    d.umur, d.suhu_badan, d.tekanan_darah_atas, d.tekanan_darah_bawah,
                    d.denyut_nadi_per_menit, d.respirasi_nafas_per_menit, d.saturasi_oksigen
                }, d.kondisi)).ToList();

                var testTuples = split80.test.Select(d => (new float[]
                {
                    d.umur, d.suhu_badan, d.tekanan_darah_atas, d.tekanan_darah_bawah,
                    d.denyut_nadi_per_menit, d.respirasi_nafas_per_menit, d.saturasi_oksigen
                }, d.kondisi)).ToList();

                knn.Fit(trainTuples);

                int correct = 0;
                foreach (var (features, label) in testTuples)
                {
                    var prediction = knn.Predict(features);
                    if (prediction == label)
                        correct++;
                }

                double acc = (double)correct / testTuples.Count;
                resultKnn.Add(($"KNN (k={k}, 80/20)", acc));
                Console.WriteLine($"   - K={k} → Akurasi: {acc:P2}");
            }
            
            {
                var knn = new KNNClassifier(k);

                var trainTuples = split90.train.Select(d => (new float[]
                {
                    d.umur, d.suhu_badan, d.tekanan_darah_atas, d.tekanan_darah_bawah,
                    d.denyut_nadi_per_menit, d.respirasi_nafas_per_menit, d.saturasi_oksigen
                }, d.kondisi)).ToList();

                var testTuples = split90.test.Select(d => (new float[]
                {
                    d.umur, d.suhu_badan, d.tekanan_darah_atas, d.tekanan_darah_bawah,
                    d.denyut_nadi_per_menit, d.respirasi_nafas_per_menit, d.saturasi_oksigen
                }, d.kondisi)).ToList();

                knn.Fit(trainTuples);

                int correct = 0;
                foreach (var (features, label) in testTuples)
                {
                    var prediction = knn.Predict(features);
                    if (prediction == label)
                        correct++;
                }

                double acc = (double)correct / testTuples.Count;
                resultKnn.Add(($"KNN (k={k}, 90/10)", acc));
                Console.WriteLine($"   - K={k} → Akurasi: {acc:P2}");
            }
        }

        var mlContext = new MLContext(seed: 0);
        var data = mlContext.Data.LoadFromEnumerable(dataList);

        // 7. Preprocessing klasifikasi berdasarkan kondisi
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

        // 8. Proses Data model untuk dibandingkan dengan metode
        var models = new Dictionary<string, IEstimator<ITransformer>>
        {
            { "Decision Tree", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree()) },
            { "Naive Bayes", mlContext.MulticlassClassification.Trainers.NaiveBayes() },
        };

        //inisialisasi temporary hasil
        var results = new List<(string Model, double MicroAcc, double MacroAcc, double LogLoss)>();

        foreach (var kvp in models)
        {
            Console.WriteLine($"\n⚙️ Processing data model: {kvp.Key}...");
            var pipeline = basePipeline.Append(kvp.Value)
                                       .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Split 80–20
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var model = pipeline.Fit(split.TrainSet);
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            results.Add((kvp.Key + " (80/20)", metrics.MicroAccuracy, metrics.MacroAccuracy, metrics.LogLoss));

            // Split 90–10
            var split2 = mlContext.Data.TrainTestSplit(data, testFraction: 0.1);
            var model2 = pipeline.Fit(split2.TrainSet);
            var predictions2 = model2.Transform(split2.TestSet);
            var metrics2 = mlContext.MulticlassClassification.Evaluate(predictions2);

            results.Add((kvp.Key + " (90/10)", metrics2.MicroAccuracy, metrics2.MacroAccuracy, metrics2.LogLoss));

            // Cross Validation
            foreach (int fold in new[] { 3, 5, 10 })
            {
                var cvResults = mlContext.MulticlassClassification.CrossValidate(data, pipeline, numberOfFolds: fold);
                var avgMicro = cvResults.Average(r => r.Metrics.MicroAccuracy);
                var avgMacro = cvResults.Average(r => r.Metrics.MacroAccuracy);
                var avgLog = cvResults.Average(r => r.Metrics.LogLoss);

                results.Add(($"{kvp.Key} (CV {fold}-Fold)", avgMicro, avgMacro, avgLog));
                Console.WriteLine($"   - {kvp.Key} {fold}-Fold → Micro: {avgMicro:P2}, Macro: {avgMacro:P2}");
            }
        }

        foreach (var (model, acc) in resultKnn)
        {
            results.Add((model, acc, acc, 0));
        }

        using (var writer = new StreamWriter("hasil_perbandingan_model.csv"))
        {
            writer.WriteLine("Model,MicroAccuracy,MacroAccuracy,LogLoss");
            foreach (var r in results)
                writer.WriteLine($"{r.Model},{r.MicroAcc},{r.MacroAcc},{r.LogLoss}");
        }

        //=== 7️. Simpan CSV & Buat Grafik ===
        using (var writer = new StreamWriter("hasil_perbandingan_model.csv"))
        {
            writer.WriteLine("Model,MicroAccuracy,MacroAccuracy,LogLoss");
            foreach (var r in results)
                writer.WriteLine($"{r.Model},{r.MicroAcc},{r.MacroAcc},{r.LogLoss}");
        }

        Console.WriteLine("\n💾 Hasil evaluasi disimpan ke: hasil_perbandingan_model.csv");

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
        
        // Plot myPlot = new();
        //
        // // create a bar plot
        // var modelNames = results.Select(r => r.Model).ToArray();
        // var accuracies = results.Select(r => r.MicroAcc * 100).ToArray();
        // double[] xs = Enumerable.Range(0, modelNames.Length).Select(i => (double)i).ToArray();
        //
        // var bars = myPlot.Add.Bars(xs, accuracies);
        // myPlot.Axes.SetLimitsY(0, accuracies.Max() + 5);
        //
        // myPlot.Axes.Left.Label.Text = "Akurasi (%)";
        // myPlot.Axes.Bottom.Label.Text = "Model";
        // myPlot.Axes.Title.Label.Text = "Perbandingan Akurasi Model Klasifikasi Vital Sign";
        //
        // Tick[] ticks = modelNames
        //     .Select((name, i) => new ScottPlot.Tick(xs[i], name))
        //     .ToArray();
        //
        // myPlot.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(ticks);
        //
        // myPlot.Axes.Bottom.TickLabelStyle.Rotation = 15;
        // myPlot.Axes.Bottom.TickLabelStyle.Alignment = Alignment.MiddleLeft;
        //
        // myPlot.Axes.Margins(bottom: 0.4, top: 0);
        //
        // myPlot.Axes.SetLimitsY(0, accuracies.Max() + 5);
        
        // myPlot.Axes.AutoScale();
        
        // myPlot.Layout.Fixed(padding: new PixelPadding(100, 40, 60, 120));

        // Simpan grafik sebagai PNG
        plt.SavePng("demo2.png", 1600, 800);
        Console.WriteLine("Grafik disimpan sebagai: grafik_perbandingan_model.png");
        Console.WriteLine("\n=== ✅ Analisis Selesai ===");
    }
}