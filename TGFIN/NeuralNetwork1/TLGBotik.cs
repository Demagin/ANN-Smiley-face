using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Types;

namespace NeuralNetwork1
{
    class TLGBotik
    {
        public Telegram.Bot.TelegramBotClient botik = null;

        private UpdateTLGMessages formUpdater;

        private BaseNetwork perseptron = null;

        GenerateImage generator = new GenerateImage();

        private Camera processor = new Camera();

        AIMLBotik abot = null;

        public TLGBotik(BaseNetwork net,  UpdateTLGMessages updater)
        { 
            var botKey = System.IO.File.ReadAllText("botkey.txt");
            botik = new Telegram.Bot.TelegramBotClient(botKey);
            botik.OnMessage += Botik_OnMessageAsync;
            formUpdater = updater;
            perseptron = net;
            abot = new AIMLBotik();
        }

        public void SetNet(BaseNetwork net)
        {
            perseptron = net;
            formUpdater("Net updated!");
        }

        private void Botik_OnMessageAsync(object sender, Telegram.Bot.Args.MessageEventArgs e)
        {
            //  Тут очень простое дело - банально отправляем назад сообщения
            var message = e.Message;
            formUpdater("Тип сообщения : " + message.Type.ToString());

            //  Получение файла (картинки)
            if (message.Type == Telegram.Bot.Types.Enums.MessageType.Photo)
            {
                formUpdater("Picture loadining started");
                var photoId = message.Photo.Last().FileId;
                File fl = botik.GetFileAsync(photoId).Result;

                var img = System.Drawing.Image.FromStream(botik.DownloadFileAsync(fl.FilePath).Result);
                
                System.Drawing.Bitmap bm = new System.Drawing.Bitmap(img);

                //  Масштабируем aforge
                var grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
                var threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();

                AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(200,200);
                var uProcessed = scaleFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(bm));
                uProcessed = grayFilter.Apply(uProcessed);
                threshldFilter.PixelBrightnessDifferenceLimit = 0.3f;
                threshldFilter.ApplyInPlace(uProcessed);
                var i = processor.OurSaveF(uProcessed);

                Sample sample = generator.GenerateButton(processor.ProcessImage2(i));

                switch (perseptron.Predict(sample))
                {
                    case SmileType.Happy: botik.SendTextMessageAsync(message.Chat.Id, "Это легко, это Happy :) !"); break;
                    case SmileType.Sad: botik.SendTextMessageAsync(message.Chat.Id, "Это легко, это Sad :( !"); break;
                    case SmileType.Surprised: botik.SendTextMessageAsync(message.Chat.Id, ":O"); break;
                    case SmileType.VHappy: botik.SendTextMessageAsync(message.Chat.Id, "Это легко, это Very Happy :D !"); break;
                    case SmileType.Wazowski: botik.SendTextMessageAsync(message.Chat.Id, "Это легко, это Wazowski о_о !"); break;
                    default: botik.SendTextMessageAsync(message.Chat.Id, "КАВО?!"); break;
                }
                formUpdater("Picture recognized!");
                return;
            }

            if (message.Text.Length > 0)
                botik.SendTextMessageAsync(message.Chat.Id, abot.Talk(message.Text));

            if (message == null || message.Type != Telegram.Bot.Types.Enums.MessageType.Text) return;
            if(message.Text == "Authors")
            {
                string authors = "В честь памяти Биб и Боб, а также одной Бубы";
                botik.SendTextMessageAsync(message.Chat.Id, "Авторы проекта : " + authors);
            }
           
            formUpdater(message.Text);
            return;
        }

        public bool Act()
        {
            try
            {
                botik.StartReceiving();
            }
            catch(Exception e) { 
                return false;
            }
            return true;
        }

        public void Stop()
        {
            botik.StopReceiving();
        }

    }
}
