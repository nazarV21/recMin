using System;
using System.ComponentModel.DataAnnotations;
using System.IO;
using TensorFlow;
using TensorFlow.Image;
using static System.Net.Mime.MediaTypeNames;

namespace ObjectDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            // Определяем параметры модели
            var imageSize = 300; // Размер изображения
            var classes = new[] { "Лазурит", "Сера", "Кварц", "Малахит", "аметист", "Амазонит" }; // Классы объектов

            // Загружаем и обрабатываем изображения
            var imageDir = @"C:\images"; // Путь к папке с изображениями
            var imageList = Directory.GetFiles(imageDir, "*.jpg"); // Получаем список файлов с изображениями
            var imageTensors = new TFTensor[imageList.Length]; // Создаем массив для хранения тензоров изображений
            var labelTensors = new TFTensor[imageList.Length]; // Создаем массив для хранения тензоров меток

            for (int i = 0; i < imageList.Length; i++)
            {
                // Загружаем изображение
                var imageBytes = File.ReadAllBytes(imageList[i]);
                var tensorImage = TFTensor.FromBuffer(imageBytes, TFTensorType.UInt8);
                var decodedImage = Image.DecodeJpeg(tensorImage, channels: 3);
                var resizedImage = Image.ResizeBilinear(decodedImage, new[] { imageSize, imageSize });
                var floatImage = Image.ConvertImageDType(resizedImage, TFDataType.Float);
                var normalizedImage = floatImage.Div(255f);

                imageTensors[i] = normalizedImage; // Добавляем тензор изображения в массив
                labelTensors[i] = TFTensor.FromBuffer(new[] { (byte)i }, TFTensorType.Int32); // Создаем тензор метки
            }

            // Создаем модель SSD
            var modelGraph = new TFGraph();
            var modelPath = @"C:\ssd_mobilenet_v2_coco_2018_03_29\frozen_inference_graph.pb"; // Путь к файлу модели
            var modelBuffer = File.ReadAllBytes(modelPath);
            modelGraph.Import(modelBuffer, "");
            var inputTensor = modelGraph["image_tensor"][0];
            var outputTensor = modelGraph["detection_boxes"][0];

            // Компилируем модель
            using (var session = new TFSession(modelGraph))
            {
                var runner = session.GetRunner();
                runner.AddInput(inputTensor, new[] { 1, imageSize, imageSize, 3 });
                runner.Fetch(outputTensor);

                // Обучаем модель
                for (int epoch = 0; epoch < 10; epoch++)
                {
                    for (int i = 0; i < imageTensors.Length; i++)
                    {
                        // Задаем входные данные
                        var input = new[] { imageTensors[i].GetArray<float>() };
                        var label = new[] { labelTensors[i].GetIntArray()[0] };
                        // Обучаем модель на текущем изображении
                        var feed = new TFTensor[] { input };
                        var target = new[] { outputTensor };
                        runner.Run(feed, target);

                        // Получаем результаты обучения
                        var output = runner.GetRunnerOutput();

                        // Преобразуем результаты в координаты прямоугольников
                        var boxes = output[0].GetValue() as float[,,,];
                        var scores = output[1].GetValue() as float[,];
                        var labels = output[2].GetValue() as float[,];

                        // Находим координаты прямоугольника с максимальной оценкой
                        var maxScore = 0f;
                        var maxIndex = 0;
                        for (int j = 0; j < scores.GetLength(0); j++)
                        {
                            if (scores[j, 0] > maxScore)
                            {
                                maxScore = scores[j, 0];
                                maxIndex = j;
                            }
                        }

                        // Получаем координаты прямоугольника и метку класса
                        var box = boxes[0, maxIndex, 0, 0];
                        var labelIndex = (int)labels[maxIndex, 0];

                        // Выводим результат
                        Console.WriteLine($"Изображение: {imageList[i]}");
                        Console.WriteLine($"Объект: {classes[labelIndex]}");
                        Console.WriteLine($"Координаты: [{box[1]}, {box[0]}, {box[3]}, {box[2]}]");
                    }
                }
            }
        }
    }
}