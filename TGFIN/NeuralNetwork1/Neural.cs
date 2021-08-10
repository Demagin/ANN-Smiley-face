using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Windows.Forms;
using System.Collections;
using System.IO;

namespace NeuralNetwork1
{
    /// <summary>
    /// Класс для хранения образа – входной массив сигналов на сенсорах, выходные сигналы сети, и прочее
    /// </summary>
    public class Sample
    {
        /// <summary>
        /// Входной вектор
        /// </summary>
        public double[] input = null;

        /// <summary>
        /// Выходной вектор, задаётся извне как результат распознавания
        /// </summary>
        public double[] output = null;

        /// <summary>
        /// Вектор ошибки, вычисляется по какой-нибудь хитрой формуле
        /// </summary>
        public double[] error = null;

        /// <summary>
        /// Действительный класс образа. Указывается учителем
        /// </summary>
        public SmileType actualClass;

        /// <summary>
        /// Распознанный класс - определяется после обработки
        /// </summary>
        public SmileType recognizedClass;

        /// <summary>
        /// Конструктор образа - на основе входных данных для сенсоров, при этом можно указать 
        /// класс образа, или не указывать
        /// </summary>
        /// <param name="inputValues"></param>
        /// <param name="sampleClass"></param>
        public Sample(double[] inputValues, int classesCount, SmileType sampleClass = SmileType.Undef)
        {
            //  Клонируем массивчик
            input = (double[]) inputValues.Clone();
            output = new double[classesCount];
            if (sampleClass != SmileType.Undef) output[(int)sampleClass] = 1;


            recognizedClass = SmileType.Undef;
            actualClass = sampleClass;
        }

        /// <summary>
        /// Обработка реакции сети на данный образ на основе вектора выходов сети
        /// </summary>
        public void processOutput()
        {
            if (error == null)
                error = new double[output.Length];
            
            //  Нам так-то выход не нужен, нужна ошибка и определённый класс
            recognizedClass = 0;
            for(int i = 0; i < output.Length; ++i)
            {
                error[i] = ((i == (int) actualClass ? 1 : 0) - output[i]);
                if (output[i] > output[(int)recognizedClass]) recognizedClass = (SmileType)i;
            }
        }

        /// <summary>
        /// Вычисленная суммарная квадратичная ошибка сети. Предполагается, что целевые выходы - 1 для верного, и 0 для остальных
        /// </summary>
        /// <returns></returns>
        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < output.Length; ++i)
                Result += Math.Pow(error[i], 2);
            return Result;
        }

        /// <summary>
        /// Добавляет к аргументу ошибку, соответствующую данному образу (не квадратичную!!!)
        /// </summary>
        /// <param name="errorVector"></param>
        /// <returns></returns>
        public void updateErrorVector(double[] errorVector)
        {
            for (int i = 0; i < errorVector.Length; ++i)
                errorVector[i] += error[i];
        }

        /// <summary>
        /// Представление в виде строки
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string result = "Sample decoding : " + actualClass.ToString() + "(" + ((int)actualClass).ToString() + "); " + Environment.NewLine + "Input : ";
            for (int i = 0; i < input.Length; ++i) result += input[i].ToString() + "; ";
            result += Environment.NewLine + "Output : ";
            if (output == null) result += "null;";
            else
                for (int i = 0; i < output.Length; ++i) result += output[i].ToString() + "; ";
            result += Environment.NewLine + "Error : ";

            if (error == null) result += "null;";
            else
                for (int i = 0; i < error.Length; ++i) result += error[i].ToString() + "; ";
            result += Environment.NewLine + "Recognized : " + recognizedClass.ToString() + "(" + ((int)recognizedClass).ToString() + "); " + Environment.NewLine;


            return result;
        }
        
        /// <summary>
        /// Правильно ли распознан образ
        /// </summary>
        /// <returns></returns>
        public bool Correct() { return actualClass == recognizedClass; }
    }
    
    /// <summary>
    /// Выборка образов. Могут быть как классифицированные (обучающая, тестовая выборки), так и не классифицированные (обработка)
    /// </summary>
    public class SamplesSet : IEnumerable
    {
        /// <summary>
        /// Накопленные обучающие образы
        /// </summary>
        public List<Sample> samples = new List<Sample>();
        
        /// <summary>
        /// Добавление образа к коллекции
        /// </summary>
        /// <param name="image"></param>
        public void AddSample(Sample image)
        {
            samples.Add(image);
        }
        public int Count { get { return samples.Count; } }

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        /// <summary>
        /// Реализация доступа по индексу
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public Sample this[int i]
        {
            get { return samples[i]; }
            set { samples[i] = value; }
        }

        public double ErrorsCount()
        {
            double correct = 0;
            double wrong = 0;
            foreach (var sample in samples)
                if (sample.Correct()) ++correct; else ++wrong;
            return correct / (correct + wrong);
        }
        // Тут бы ещё сохранение в файл и чтение сделать, вообще классно было бы
    }



    public class NeuralNetwork : BaseNetwork
    {
        ///  Ваш код здесь
        /// <summary>
        /// Реализация кастомной нейронной сети
        /// </summary>
        private MyActivationNetwork network = null;

        /// <summary>
        /// Значение ошибки при обучении единичному образцу. При обучении на наборе образов не используется
        /// </summary>
        public double desiredErrorValue = 0.0005;

        //  Секундомер спортивный, завода «Агат», измеряет время пробегания стометровки, ну и время затраченное на обучение тоже умеет
        public System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();

        /// <summary>
        /// Конструктор сети с указанием структуры (количество слоёв и нейронов в них)
        /// </summary>
        /// <param name="structure"></param>
        public NeuralNetwork(int[] structure)
        {
            ReInit(structure);
        }

        /// <summary>
        /// Реиницализация сети - создаём заново сеть с указанной структурой
        /// </summary>
        /// <param name="structure">Массив с указанием нейронов на каждом слое (включая сенсорный)</param>
        /// <param name="initialLearningRate">Начальная скорость обучения</param>
        public override void ReInit(int[] structure, double initialLearningRate = 0.25)
        {
            // Создаём сеть - вроде того
            network = new MyActivationNetwork((MyActivationFunction)new MySigmoidFunction(), structure[0],
                structure.Skip(1).ToArray<int>());

            //  Встряска "мозгов" сети - рандомизируем веса связей
                network.Randomize();
        }

        /// <summary>
        /// Обучение сети одному образу
        /// </summary>
        /// <param name="sample"></param>
        /// <returns>Количество итераций для достижения заданного уровня ошибки</returns>
        public override int Train(Sample sample, bool parallel = true)
        {
            // Создаём "обучателя" - либо параллельного, либо последовательного  
            MyResilientBackpropagationLearning teacher;
            if (parallel)
                teacher = new MyResilientBackpropagationLearning(network);
            else
               teacher = new MyResilientBackpropagationLearning(network);

            int iters = 1;
            while (teacher.Run(sample.input, sample.output) > desiredErrorValue) { ++iters; }
            return iters;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_erorr, bool parallel = true)
        {
            //  Сначала надо сконструировать массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            //  Теперь массивы из samplesSet группируем в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].output;
            }

            //  Текущий счётчик эпох
            int epoch_to_run = 0;

            //  Создаём "обучателя" - либо параллельного, либо последовательного  
            MyResilientBackpropagationLearning teacher;
            if (parallel)
                teacher = new MyResilientBackpropagationLearning(network);
            else
                teacher = new MyResilientBackpropagationLearning(network);

            double error = double.PositiveInfinity;

#if DEBUG
            StreamWriter errorsFile = File.CreateText("errors.csv");
#endif

            stopWatch.Restart();

            while (epoch_to_run < epochs_count && error > acceptable_erorr)
            {
                epoch_to_run++;
                error = teacher.RunEpoch(inputs, outputs);
#if DEBUG
                errorsFile.WriteLine(error);
#endif
                updateDelegate((epoch_to_run * 1.0) / epochs_count, error, stopWatch.Elapsed);
            }

#if DEBUG
            errorsFile.Close();
#endif

            updateDelegate(1.0, error, stopWatch.Elapsed);

            stopWatch.Stop();

            return error;
        }

        public override SmileType Predict(Sample sample)
        {
            sample.output = network.Compute(sample.input);
            sample.processOutput();
            return sample.recognizedClass;
        }

        public override double TestOnDataSet(SamplesSet testSet)
        {
            double correct = 0.0;
            for (int i = 0; i < testSet.Count; ++i)
            {
                testSet[i].output = network.Compute(testSet[i].input);
                testSet[i].processOutput();
                if (testSet[i].actualClass == testSet[i].recognizedClass) correct += 1;
            }
            return correct / testSet.Count;
        }

        public override double[] getOutput()
        {
            return network.output;
        }

    }
}
