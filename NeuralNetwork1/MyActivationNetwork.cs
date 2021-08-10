using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class MyActivationNetwork
    {
        public int inputsCount;//Количество входов в сеть
        public int layersCount;//Количество слоев сети
        public MyLayer[] layers; //Слои сети
        public double[] output;//Выходной вектор сети

        public MyActivationNetwork(MyActivationFunction f, int inputsC, int[] neuronsC)
        {
            this.inputsCount = Math.Max(1, inputsC);
            this.layersCount = Math.Max(1, neuronsC.Length);
            this.layers = new MyLayer[neuronsC.Length];
            this.output = new double[neuronsC.Length];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = new MyLayer(neuronsC[i], (i == 0) ? inputsCount : neuronsC[i - 1], f);
        }

        public double[] Compute(double[] input)
        {
            double[] output = input;
            for (int i = 0; i < layers.Length; i++)
                output = layers[i].Compute(output);
            this.output = output;
            return output;
        }

        public virtual void Randomize()
        {
            foreach (var l in layers)
                l.Randomize();
        }
    }
}
