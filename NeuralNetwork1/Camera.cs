using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Video;
using AForge.Video.DirectShow;
using AForge.Imaging.Filters;
using System.Drawing;

namespace NeuralNetwork1
{
    class Camera
    {
        private readonly object balanceLock = new object();
        // обработанное изображение 
        public AForge.Imaging.UnmanagedImage processed;
        // оригинал и финальное
        public Bitmap original, number;
        public int BlobCount { get; private set; }
        public bool Recongnised { get; private set; }
        public float Angle { get; private set; }
        public float AngleRad { get; private set; }
        public double gravit;
        public float ThresholdValue = 0.3f;


        Grayscale grayFilter;
        ResizeBilinear scaleFilter;
        BradleyLocalThresholding threshldFilter;
        Invert InvertFilter;
        AForge.Imaging.BlobCounter Blober;

        Graphics g;

        public Camera()
        {
            grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
            scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(200, 200);
            threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();
            InvertFilter = new AForge.Imaging.Filters.Invert();
            Blober = new AForge.Imaging.BlobCounter();
            original = new Bitmap(200, 200);
            g = Graphics.FromImage(original);
            Blober.FilterBlobs = true;
            Blober.MinWidth = 5;
            Blober.MinHeight = 5;
            Blober.ObjectsOrder = AForge.Imaging.ObjectsOrder.Size;
        }

        public void GetPicture(out Bitmap n)
        {
            lock(balanceLock)
            {
                n = new Bitmap(number);
            }
        }

        public void ProcessImage(Bitmap input_image)
        {
            lock (balanceLock)
            {
                int side = Math.Min(input_image.Height, input_image.Width);
                Rectangle cropRect = new Rectangle(0, 0, side, side); 
                g.DrawImage(input_image, new Rectangle(0, 0, input_image.Width, input_image.Height), cropRect, GraphicsUnit.Pixel);      
                                                                                                                                    
                if (processed != null)
                    processed.Dispose();  
                
                //  Конвертируем изображение в градации серого
                processed = grayFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));
                //  Пороговый фильтр применяем. Величина порога берётся из настроек, и меняется на форме
                threshldFilter.PixelBrightnessDifferenceLimit = ThresholdValue;
                threshldFilter.ApplyInPlace(processed);
                Blober.ProcessImage(processed);
                AForge.Imaging.Blob[] blobs = Blober.GetObjectsInformation();
                BlobCount = blobs.Length;

                if (number != null)
                    number.Dispose();
                number = processed.ToManagedImage();

            }

        }

        public Bitmap OurSave()
        {
            var processed1 = processed.Clone();

            AForge.Imaging.BlobCounter blober;
            blober = new AForge.Imaging.BlobCounter();
            blober.FilterBlobs = true;
            blober.MinWidth = 5;
            blober.MinHeight = 5;
            blober.ObjectsOrder = AForge.Imaging.ObjectsOrder.Size;
            InvertFilter.ApplyInPlace(processed1);
            blober.ProcessImage(processed1);

            AForge.Imaging.Blob[] blobs = blober.GetObjectsInformation();

            Rectangle[] rects = blober.GetObjectsRectangles();
            
            // К сожалению, код с использованием подсчёта blob'ов не работает, поэтому просто высчитываем максимальное покрытие
            // для всех блобов - для нескольких цифр, к примеру, 16, можем получить две области - отдельно для 1, и отдельно для 6.
            // Строим оболочку, включающую все блоки. Решение плохое, требуется доработка
            int lx = processed1.Width;
            int ly = processed1.Height;
            int rx = 0;
            int ry = 0;
            for (int i = 0; i < rects.Length; ++i)
            {
                if (lx > rects[i].X) lx = rects[i].X;
                if (ly > rects[i].Y) ly = rects[i].Y;
                if (rx < rects[i].X + rects[i].Width) rx = rects[i].X + rects[i].Width;
                if (ry < rects[i].Y + rects[i].Height) ry = rects[i].Y + rects[i].Height;
            }

            // Обрезаем края, оставляя только центральные блобчики
            AForge.Imaging.Filters.Crop cropFilter = new AForge.Imaging.Filters.Crop(new Rectangle(lx, ly, rx - lx, ry - ly));  
            processed1 = cropFilter.Apply(processed1);

            //  Масштабируем до 100x100
            AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(100, 100);
            processed1 = scaleFilter.Apply(processed1);

            return processed1.ToManagedImage();
        }

        public double[] ProcessImage2(Bitmap input_image)
        {
            double angle = 0.0;
            double anglerad = 0.0;
            double range = 0.0;

            AForge.Imaging.BlobCounter blober;
            blober = new AForge.Imaging.BlobCounter();
            blober.FilterBlobs = true;
            blober.MinWidth = 5;
            blober.MinHeight = 5;
            blober.ObjectsOrder = AForge.Imaging.ObjectsOrder.Size;


            Bitmap original2 = new Bitmap(200, 200);
            Graphics g2 = Graphics.FromImage(original2);


            int side = Math.Min(input_image.Height, input_image.Width);
            Rectangle cropRect = new Rectangle(0, 0, side, side); 
            g2.DrawImage(input_image, new Rectangle(0, 0, input_image.Width, input_image.Height), cropRect, GraphicsUnit.Pixel);      

            AForge.Imaging.UnmanagedImage cameraman;
            //  Конвертируем изображение в градации серого
            cameraman = AForge.Imaging.UnmanagedImage.FromManagedImage(original2);
            blober.ProcessImage(cameraman);
            AForge.Imaging.Blob[] blobs = blober.GetObjectsInformation();

            int recX = blobs[1].Rectangle.X;
            int recY = blobs[1].Rectangle.Y;
            double fullness = blobs[1].Fullness;

            if (blobs.Length > 0)
            {
                var MouthBlob = blobs[1];
                blober.ExtractBlobsImage(cameraman, MouthBlob, false);
                AForge.Point mc = MouthBlob.CenterOfGravity;
                AForge.Point ic = new AForge.Point((float)MouthBlob.Image.Width / 2, (float)MouthBlob.Image.Height / 2);
                anglerad = (ic.Y - mc.Y) / (ic.X - mc.X);
                angle = (float)(Math.Atan(anglerad) * 180 / Math.PI);

                var minBlob = blobs[blobs.Length - 1];
                blober.ExtractBlobsImage(cameraman, minBlob, false);
                AForge.Point mc2 = minBlob.CenterOfGravity;
                //AB = √(xb - xa)2 + (yb - ya)2
                range = Math.Sqrt(Math.Pow(mc.X - mc2.X, 2) + Math.Pow(mc.Y - mc2.Y, 2));
            }
            else
            {
                angle = 0;
                range = 0.0;
            }

            return new double[5] {angle, recX, recY, fullness, range};

        }

    }
}
