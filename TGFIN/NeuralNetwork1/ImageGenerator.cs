using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    /// <summary>
    /// Тип фигуры
    /// </summary>
    public enum FigureType : byte { Triangle = 0, Rectangle, Circle, Sinusiod, Undef };

    public enum SmileType : byte { Happy = 0, Sad, Surprised, VHappy, Wazowski, Undef };

    public class GenerateImage
    {
        /// <summary>
        /// Бинарное представление образа
        /// </summary>
        public bool[,] img = new bool[200, 200]; 
        
        /// <summary>
        /// Текущая сгенерированная фигура
        /// </summary>
        public SmileType current_Button = SmileType.Undef;

        /// <summary>
        /// Количество классов генерируемых фигур (5 - максимум)
        /// </summary>
        public int figure_count { get; set; } = 5;
       
        /// <summary>
        /// Очистка образа
        /// </summary>
        public void ClearImage()
        {
            for (int i = 0; i < 200; ++i)
                for (int j = 0; j < 200; ++j)
                    img[i, j] = false;
        }

        public Sample GenerateButton(double[] input, int Button = 0)
        {
            current_Button = (SmileType)Button;

            SmileType type = current_Button;

            return new Sample(input, figure_count, current_Button);
        }

        /// <summary>
        /// Возвращает битовое изображение для вывода образа
        /// </summary>
        /// <returns></returns>
        public Bitmap genBitmap()
        {
            Bitmap DrawArea = new Bitmap(200, 200);
            for (int i = 0; i < 200; ++i)
                for (int j = 0; j < 200; ++j)
                    if (img[i, j])
                        DrawArea.SetPixel(i, j, Color.Black);
            return DrawArea;
        }
    }

}
