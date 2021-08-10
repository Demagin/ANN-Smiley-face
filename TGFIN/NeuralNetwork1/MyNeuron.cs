using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class MyNeuron
    {
        public int inputsCount;//Количество входов нейрона
        public double[] weights;//Веса нейрона
        public double output; //Выходное значение нейрона
        public Random rand = new Random();// Генератор случайных чисел.
        public double threshold = 0.0;

        public MyActivationFunction function;

        public MyNeuron(int inputsC, MyActivationFunction f)
        {
            inputsCount = Math.Max(1, inputsC);
            weights = new double[inputsCount];
            function = f;
        }

        public virtual void Randomize()
        {
            for (int i = 0; i < weights.Length; i++)
                weights[i] = -rand.NextDouble() / 10f + rand.NextDouble() / 10f;
            threshold = rand.NextDouble();
        }

        public double Compute(double[] input)
        {
            double sum = 0.0;
            for (int i = 0; i < weights.Length; i++)
                sum += weights[i] * input[i];
            sum += threshold;
            double output = function.Function(sum);
            this.output = output;

            return output;
        }
    }
}


