using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    interface MyActivationFunction
    {
        double Function(double x);//вычисляет значение функции

        double Derivative(double y);//вычисляет производную
    }

    //класс функции сигмоиды
    class MySigmoidFunction : MyActivationFunction
    {
        public MySigmoidFunction() { }

        public double Function(double x)
        {
            return (1 / (1 + Math.Exp(-x)));
        }

        public double Derivative(double y)
        {
            return (y * (1 - y));
        }
    }
}
