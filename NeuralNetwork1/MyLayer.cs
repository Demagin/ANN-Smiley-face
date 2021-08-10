using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class MyLayer
    {
        public int inputsCount;// Количество входов слоя
        public int neuronsCount;// Слои нейронов считаются
        public MyNeuron[] neurons;// Слои нейронов
        public double[] output;// Выходной вектор слоя

        public MyLayer(int neuronsCnt, int inputsCnt, MyActivationFunction f)
        {
            inputsCount = Math.Max(1, inputsCnt);
            neuronsCount = Math.Max(1, neuronsCnt);
            neurons = new MyNeuron[neuronsCnt];
            for (int i = 0; i < neurons.Length; i++)
                neurons[i] = new MyNeuron(inputsCount, f);
            output = new double[neuronsCnt];
        }

        public double[] Compute(double[] input)
        {
            double[] output = new double[neuronsCount];
            for (int i = 0; i < neurons.Length; i++)
                output[i] = neurons[i].Compute(input);
            this.output = output;

            return output;
        }

        public virtual void Randomize()
        {
            foreach (var neuron in neurons)
                neuron.Randomize();
        }
    }
}
