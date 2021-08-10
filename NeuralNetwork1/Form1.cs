using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using AForge.Video;
using AForge.Video.DirectShow;

namespace NeuralNetwork1
{

	public delegate void FormUpdater(double progress, double error, TimeSpan time);

    public partial class Form1 : Form
    {
        private IVideoSource videoSource;
        private FilterInfoCollection videoDevicesList;
        private Camera processor = new Camera();

        /// <summary>
        /// Генератор изображений (образов)
        /// </summary>
        GenerateImage generator = new GenerateImage();
        
        /// <summary>
        /// Самодельный персептрон – из массивов и палок
        /// </summary>
        NeuralNetwork CustomNet = null;

        /// <summary>
        /// Обёртка для ActivationNetwork из Accord.Net
        /// </summary>
        AccordNet AccordNet = null;

        /// <summary>
        /// Абстрактный базовый класс, псевдоним либо для CustomNet, либо для AccordNet
        /// </summary>
        BaseNetwork net = null;
        string imgFile = "";
        public Form1()
        {
            InitializeComponent();
            netTypeBox.SelectedIndex = 1;
            generator.figure_count = (int)classCounter.Value;
            button3_Click(this, null);
            pictureBox1.Image = Properties.Resources.Title;

            videoDevicesList = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            foreach (FilterInfo videoDevice in videoDevicesList)
            {
                cmbVideoSource.Items.Add(videoDevice.Name);
            }
            if (cmbVideoSource.Items.Count > 0)
            {
                cmbVideoSource.SelectedIndex = 1;
            }
            else
            {
                MessageBox.Show("Камера не найдена!", "Ошибка!", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            CloseOpenVideoSource();

            for (int i = 0; i < 5; i++)
                KindOfObjextComboBox.Items.Add((SmileType)i);

            KindOfObjextComboBox.SelectedIndex = 0;
            imgFile = System.IO.Directory.GetCurrentDirectory() + "\\";
        }

		public void UpdateLearningInfo(double progress, double error, TimeSpan elapsedTime)
		{
			if (progressBar1.InvokeRequired)
			{
				progressBar1.Invoke(new FormUpdater(UpdateLearningInfo),new Object[] {progress, error, elapsedTime});
				return;
			}
            StatusLabel.Text = "Accuracy: " + error.ToString();
            int prgs = (int)Math.Round(progress*100);
			prgs = Math.Min(100, Math.Max(0,prgs));
            elapsedTimeLabel.Text = "Затраченное время : " + elapsedTime.Duration().ToString(@"hh\:mm\:ss\:ff");
            progressBar1.Value = prgs;
		}


        private void set_result(Sample figure)
        {
            label1.Text = figure.ToString();

            label1.Text = "Распознано : " + (figure.recognizedClass);

            label8.Text = String.Join("\n", net.getOutput().Select(d => d.ToString()));
            pictureBox1.Image = generator.genBitmap();
            pictureBox1.Invalidate();
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            Bitmap img = processor.OurSave();
            Sample fig = generator.GenerateButton(img, processor.ProcessImage2(img));
            net.Predict(fig);

            set_result(fig);
        }

        private async Task<double> train_networkAsync( int epoches, double acceptable_error, bool parallel = true)
        {

            //  Выключаем всё ненужное
            label1.Text = "Выполняется обучение...";
            label1.ForeColor = Color.Red;
            groupBox1.Enabled = false;
            pictureBox1.Enabled = false;
            trainOneButton.Enabled = false;

            //  Создаём новую обучающую выборку
            SamplesSet samples = new SamplesSet();

            int smp = 0;

            for (int i = 0; i < (int)classCounter.Value; i++)
            {
                int cnt = new System.IO.DirectoryInfo(imgFile + "\\" + (SmileType)smp).GetFiles().Length;
                for (int j = 0; j < cnt - 1; j++)
                {
                    string s = imgFile + (SmileType)smp + "\\" + (SmileType)smp + (j+1).ToString() + ".bmp";
                    Bitmap img = new Bitmap(Image.FromFile(s));
                    var v = processor.ProcessImage2((Bitmap)img);
                    Sample fig = generator.GenerateButton(img, processor.ProcessImage2((Bitmap)img), smp);
                    textBox1.Text += "НОВАЯ"+v[0]+ " "+ v[1]+ " "+ v[2]+ " " + v[3] + " " + "\n";
                    samples.AddSample(fig);
                }
                smp++;
            }

            //  Обучение запускаем асинхронно, чтобы не блокировать форму
            double f = await Task.Run(() => net.TrainOnDataSet(samples, epoches, acceptable_error, parallel));

            label1.Text = "Щелкните на картинку для теста нового образа";
            label1.ForeColor = Color.Green;
            groupBox1.Enabled = true;
            pictureBox1.Enabled = true;
            trainOneButton.Enabled = true;
            StatusLabel.Text = "Accuracy: " + f.ToString();
            StatusLabel.ForeColor = Color.Green;
            return f;
            
        }

        private void button1_Click(object sender, EventArgs e)
        {

            #pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
            train_networkAsync((int)EpochesCounter.Value, (100 - AccuracyCounter.Value) / 100.0, parallelCheckBox.Checked);
            #pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        }

        private void button3_Click(object sender, EventArgs e)
        {
            //  Проверяем корректность задания структуры сети
            int[] structure = netStructureBox.Text.Split(';').Select((c) => int.Parse(c)).ToArray();
            //if (structure.Length < 4 || structure[0] != 400 || structure[structure.Length - 1] != generator.figure_count)
            //{
            //    MessageBox.Show("А давайте вы структуру сети нормально запишите, ОК?", "Ошибка", MessageBoxButtons.OK);
            //    return;
            //};

            CustomNet = new NeuralNetwork(structure);
            CustomNet.updateDelegate = UpdateLearningInfo;

            AccordNet = new AccordNet(structure);
            AccordNet.updateDelegate = UpdateLearningInfo;

            if (netTypeBox.SelectedIndex == 0)
                net = CustomNet;
            else
                net = AccordNet;
        }

        private void classCounter_ValueChanged(object sender, EventArgs e)
        {
            generator.figure_count = (int)classCounter.Value;
            var vals = netStructureBox.Text.Split(';');
            int outputNeurons;
            if (int.TryParse(vals.Last(),out outputNeurons))
            {
                vals[vals.Length - 1] = classCounter.Value.ToString();
                netStructureBox.Text = vals.Aggregate((partialPhrase, word) => $"{partialPhrase};{word}");
            }
        }

        private void btnTrainOne_Click(object sender, EventArgs e)
        {
            if (net == null) return;
            int cnt = new System.IO.DirectoryInfo(imgFile + "\\" + KindOfObjextComboBox.SelectedItem).GetFiles().Length;
            Bitmap img = new Bitmap(Image.FromFile(imgFile + KindOfObjextComboBox.SelectedItem + "\\" + KindOfObjextComboBox.SelectedItem + cnt.ToString() + ".bmp"));
            var v = processor.ProcessImage2((Bitmap)img);
            Sample fig = generator.GenerateButton(img, processor.ProcessImage2((Bitmap)img),  KindOfObjextComboBox.SelectedIndex) as Sample;
            pictureBox1.Image = generator.genBitmap();
            pictureBox1.Invalidate();
            net.Train(fig, false);
            set_result(fig);
        }

        private void netTrainButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Обучить нейросеть с указанными параметрами";
        }

        private void netTypeBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (netTypeBox.SelectedIndex == 0)
                net = CustomNet;
            else
                net = AccordNet;
        }

        private void recreateNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Заново пересоздаёт сеть с указанными параметрами";
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            button1.Enabled = false;
            int cnt = new System.IO.DirectoryInfo(imgFile + "\\" + KindOfObjextComboBox.SelectedItem).GetFiles().Length;
            

            Bitmap b = new Bitmap(processor.number.Width, processor.number.Height);
            b = processor.number;
            b = processor.OurSave();
            cnt++;
            string s = imgFile + KindOfObjextComboBox.SelectedItem + "\\" + KindOfObjextComboBox.SelectedItem + cnt.ToString() + ".bmp";
            b.Save(s);

            button1.Enabled = true;
        }

        private void video_NewFrame(object sender, NewFrameEventArgs eventArgs)
        {
          
            processor.ProcessImage((Bitmap)eventArgs.Frame.Clone());
            processor.GetPicture(out Bitmap n);

            if (processor.number != null)
                pictureBox1.Image = n;
        }

        void CloseOpenVideoSource()
        {
            if (videoSource == null)
            {
                videoSource = new VideoCaptureDevice(videoDevicesList[cmbVideoSource.SelectedIndex].MonikerString);
                videoSource.NewFrame += new NewFrameEventHandler(video_NewFrame);
                videoSource.Start();
                btnStart.Text = "Stop";
            }
            else
            {
                videoSource.SignalToStop();
                videoSource = null;
                btnStart.Text = "Start";

            }

        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            CloseOpenVideoSource();
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (btnStart.Text == "Stop")
                videoSource.SignalToStop();
            videoSource = null;
        }

    }

  }
