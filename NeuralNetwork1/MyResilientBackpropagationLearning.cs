using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class MyResilientBackpropagationLearning
    {
        private MyActivationNetwork network;

        private double learningRate = 0.0125;//начальная скорость обучения
        private double deltaMax = 50.0;//потолок для дельты
        private double deltaMin = 1e-6;//пол для дельты

        private const double etaMinus = 0.5;//η для уменьшения веса
        private double etaPlus = 1.2;//η для увеличения веса

        private double[][] neuronErrors = null;//ошкибка каждого нейрона

        //дельты на которые будем обновлять веса, вначале = learningRate
        private double[][][] weightsUpdates = null;
        private double[][] thresholdsUpdates = null;

        // текущие и предыдущие значения градиента, на сколько каждый вес влияет на ошибку нейрона
        private double[][][] weightsDerivatives = null;
        private double[][] thresholdsDerivatives = null;

        private double[][][] weightsPreviousDerivatives = null;
        private double[][] thresholdsPreviousDerivatives = null;

        public double LearningRate
        {
            get { return learningRate; }
            set
            {
                learningRate = value;
                ResetUpdates(learningRate);
            }
        }

        public MyResilientBackpropagationLearning(MyActivationNetwork network)
        {
            this.network = network;

            int layersCount = network.layers.Length;

            //инициализируем все массивы
            neuronErrors = new double[layersCount][];

            weightsDerivatives = new double[layersCount][][];
            thresholdsDerivatives = new double[layersCount][];

            weightsPreviousDerivatives = new double[layersCount][][];
            thresholdsPreviousDerivatives = new double[layersCount][];

            weightsUpdates = new double[layersCount][][];
            thresholdsUpdates = new double[layersCount][];

            //для каждого слоя
            for (int i = 0; i < network.layers.Length; i++)
            {
                MyLayer layer = network.layers[i];//текущий слой

                int neuronsCount = layer.neurons.Length;

                neuronErrors[i] = new double[neuronsCount];

                weightsDerivatives[i] = new double[neuronsCount][];
                weightsPreviousDerivatives[i] = new double[neuronsCount][];
                weightsUpdates[i] = new double[neuronsCount][];

                thresholdsDerivatives[i] = new double[neuronsCount];
                thresholdsPreviousDerivatives[i] = new double[neuronsCount];
                thresholdsUpdates[i] = new double[neuronsCount];

                for (int j = 0; j < layer.neurons.Length; j++)
                {
                    weightsDerivatives[i][j] = new double[layer.inputsCount];
                    weightsPreviousDerivatives[i][j] = new double[layer.inputsCount];
                    weightsUpdates[i][j] = new double[layer.inputsCount];
                }
            }

            //инициализирует дельты начальной скоростью обучения
            ResetUpdates(learningRate);
        }
        public double Run(double[] input, double[] output)
        {
            // нулевой градиент
            ResetGradient();

            // вычисляем результат сети на данном образе
            network.Compute(input);

            // рассчитываем сетевую ошибку
            double error = CalculateError(output);

            // пересчитываем веса
            CalculateGradient(input);

            // обновляем сеть
            UpdateNetwork();

            // возвращаем итоговую ошибку сети
            return error;
        }

        public double RunEpoch(double[][] input, double[][] output)
        {
            // нулевой градиент
            ResetGradient();

            double error = 0.0;

            // обучаем на каждом на каждом образе
            for (int i = 0; i < input.Length; i++)
            {
                // вычисляем результат сети на текущем образе
                network.Compute(input[i]);

                // рассчитываем ошибку сети
                error += CalculateError(output[i]);

                // пересчитываем веса
                CalculateGradient(input[i]);
            }

            // обновляем сеть
            UpdateNetwork();

            // возвращаем итоговую ошибку сети
            return error;
        }

        //Сбрасываем градиент
        private void ResetGradient()
        {
            for (int i = 0; i < weightsDerivatives.Length; i++)
                for (int j = 0; j < weightsDerivatives[i].Length; j++)
                    Array.Clear(weightsDerivatives[i][j], 0, weightsDerivatives[i][j].Length);

            for (int i = 0; i < thresholdsDerivatives.Length; i++)
                 Array.Clear(thresholdsDerivatives[i], 0, thresholdsDerivatives[i].Length);
        }

        //сбрасываем дельты, по умолчанию начальная скорость
        private void ResetUpdates(double rate)
        {
            for (int i = 0; i < weightsUpdates.Length; i++)
                for (int j = 0; j < weightsUpdates[i].Length; j++)
                    for (int k = 0; k < weightsUpdates[i][j].Length; k++)
                        weightsUpdates[i][j][k] = rate;                                             

            for (int i = 0; i < thresholdsUpdates.Length; i++)  
                for (int j = 0; j < thresholdsUpdates[i].Length; j++)     
                    thresholdsUpdates[i][j] = rate;
                
            
        }

        //обновляем веса сети
        private void UpdateNetwork()
        {
            //ссылки на массивы чтобы уменшить количество циклов и индексов
            double[][] layerweightsUpdates;
            double[] layerthresholdUpdates;
            double[] neuronWeightUpdates;

            double[][] layerweightsDerivatives;
            double[] layerthresholdDerivatives;
            double[] neuronWeightDerivatives;

            double[][] layerPreviousweightsDerivatives;
            double[] layerPreviousthresholdDerivatives;
            double[] neuronPreviousWeightDerivatives;

            //идем по каждому слою сети
            for (int i = 0; i < network.layers.Length; i++)
            {
                MyLayer layer = network.layers[i];//текущий слой

                //ссылки на массивы чтобы уменшить количество циклов и индексов
                layerweightsUpdates = weightsUpdates[i];//дельты весов нейронов текущего слоя
                layerthresholdUpdates = thresholdsUpdates[i];//дельты порогов нейронов текущего слоя

                layerweightsDerivatives = weightsDerivatives[i];
                layerthresholdDerivatives = thresholdsDerivatives[i];

                layerPreviousweightsDerivatives = weightsPreviousDerivatives[i];
                layerPreviousthresholdDerivatives = thresholdsPreviousDerivatives[i];

                //идем по каждом нейрону сети
                for (int j = 0; j < layer.neurons.Length; j++)
                {
                    MyNeuron neuron = layer.neurons[j];//текущий нейрон

                    neuronWeightUpdates = layerweightsUpdates[j];//дельты весов для текущего нейрона
                    neuronWeightDerivatives = layerweightsDerivatives[j];//текущие значения градиента для текущего нейрона
                    neuronPreviousWeightDerivatives = layerPreviousweightsDerivatives[j];//предыдущие значения градиента для текущего нейрона

                    double S = 0;

                    //для каждого входа нейрона
                    for (int k = 0; k < neuron.inputsCount; k++)
                    {
                        S = neuronPreviousWeightDerivatives[k] * neuronWeightDerivatives[k];

                        if (S > 0)
                        {
                            neuronWeightUpdates[k] = Math.Min(neuronWeightUpdates[k] * etaPlus, deltaMax);
                            neuron.weights[k] -= Math.Sign(neuronWeightDerivatives[k]) * neuronWeightUpdates[k];
                            neuronPreviousWeightDerivatives[k] = neuronWeightDerivatives[k];
                        }
                        else if (S < 0)
                        {
                            neuronWeightUpdates[k] = Math.Max(neuronWeightUpdates[k] * etaMinus, deltaMin);
                            neuronPreviousWeightDerivatives[k] = 0;
                        }
                        else
                        {
                            neuron.weights[k] -= Math.Sign(neuronWeightDerivatives[k]) * neuronWeightUpdates[k];
                            neuronPreviousWeightDerivatives[k] = neuronWeightDerivatives[k];
                        }
                    }

                    S = layerPreviousthresholdDerivatives[j] * layerthresholdDerivatives[j];

                    if (S > 0)
                    {
                        layerthresholdUpdates[j] = Math.Min(layerthresholdUpdates[j] * etaPlus, deltaMax);
                        neuron.threshold -= Math.Sign(layerthresholdDerivatives[j]) * layerthresholdUpdates[j];
                        layerPreviousthresholdDerivatives[j] = layerthresholdDerivatives[j];
                    }
                    else if (S < 0)
                    {
                        layerthresholdUpdates[j] = Math.Max(layerthresholdUpdates[j] * etaMinus, deltaMin);
                        layerthresholdDerivatives[j] = 0;
                    }
                    else
                    {
                        neuron.threshold -= Math.Sign(layerthresholdDerivatives[j]) * layerthresholdUpdates[j];
                        layerPreviousthresholdDerivatives[j] = layerthresholdDerivatives[j];
                    }
                }
            }
        }

        //считаем ошибки на каждом слое и возвращает ошибку сети
        private double CalculateError(double[] desiredOutput)
        {
            double error = 0;//ошибка сети
            int layersCount = network.layers.Length;//количество слоев в сети

            //функция активации, у нас всегда сигмоида
            MyActivationFunction function = (network.layers[0].neurons[0] as MyNeuron).function;

            //считаем ошибки выходного слоя сети
            MyLayer layer = network.layers[layersCount - 1];//выходной слой сети
            double[] layerDerivatives = neuronErrors[layersCount - 1];//на один массив указывает, ссылка с новым именем для удобства

            //проходимся по всем нейронам слоя и считаем ошибки
            for (int i = 0; i < layer.neurons.Length; i++)
            {
                double output = layer.neurons[i].output;//выходное значение нейрона

                double e = desiredOutput[i] - output;//желаемое значение минус полученное
                layerDerivatives[i] = e * function.Derivative(output);// δyj = j(1-yj)(dj-yj) формула ошибки, производная на разницу желаемого и полученного
                error += (e * e);//добавляем к суммарной ошибке слоя
            }

            //считаем ошибки для других слоев
            for (int j = layersCount - 2; j >= 0; j--)
            {
                layer = network.layers[j];//j-тый слой
                layerDerivatives = neuronErrors[j];//ошибки нейронов j-того слоя δ

                MyLayer layerNext = network.layers[j + 1];//j+1-тый слой
                double[] nextDerivatives = neuronErrors[j + 1];//ошибки нейронов следующего слоя δ

                //проходимся по всем нейронам слоя
                for (int i = 0, n = layer.neurons.Length; i < n; i++)
                {
                    double sum = 0.0;

                    //идем по все нейронам следюущего слоя
                    for (int k = 0; k < layerNext.neurons.Length; k++)//∑δjWjk умножаем ошибку нейрона из следующего слоя на вес, который повлиял на эту ошибку и суммируем
                        sum += nextDerivatives[k] * layerNext.neurons[k].weights[i];//yj(1-yj)∑δjWjk умножаем на производную

                    layerDerivatives[i] = sum * function.Derivative(layer.neurons[i].output);
                }
            }

            //возвращаем ошибку сети
            return error;
        }

        private void CalculateGradient(double[] input)
        {
            MyLayer layer = network.layers[0];//первый слой
            double[] weightErrors = neuronErrors[0];
            double[][] layerweightsDerivatives = weightsDerivatives[0];
            double[] layerthresholdDerivatives = thresholdsDerivatives[0];

            //считаем первый слой
            for (int i = 0; i < layer.neurons.Length; i++)
            {
                MyNeuron neuron = layer.neurons[i];
                double[] neuronWeightDerivatives = layerweightsDerivatives[i];

                //проходимся по всем входящим весам нейрона
                for (int j = 0; j < neuron.inputsCount; j++)
                    neuronWeightDerivatives[j] += weightErrors[i] * input[j];
                layerthresholdDerivatives[i] += weightErrors[i];
            }

            //идем по остальным слоям
            for (int k = 1; k < network.layers.Length; k++)
            {
                layer = network.layers[k];
                weightErrors = neuronErrors[k];
                layerweightsDerivatives = weightsDerivatives[k];
                layerthresholdDerivatives = thresholdsDerivatives[k];

                MyLayer layerPrev = network.layers[k - 1];

                //для каждого нейрона слоя
                for (int i = 0; i < layer.neurons.Length; i++)
                {
                    double[] neuronWeightDerivatives = layerweightsDerivatives[i];

                    //проходимся по всем входящим весам нейрона
                    for (int j = 0; j < layerPrev.neurons.Length; j++)
                        neuronWeightDerivatives[j] += weightErrors[i] * layerPrev.neurons[j].output;
                    layerthresholdDerivatives[i] += weightErrors[i];
                }
            }
        }
    }
}

