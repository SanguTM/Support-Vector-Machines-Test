using System;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
// using Accord.Statistics.Distances;

// uses Accord.MachineLearning v3.8.0
// Framework 4.7
//https://visualstudiomagazine.com/Articles/2019/02/01/Support-Vector-Machines.aspx?Page=1
namespace SVM_CSharp
{
    class SVM_Program
    {
        static void Main(string[] args)
        {
            //Console.WriteLine("\nBegin Support Vector Machine demo \n");

            // Paruošiami testiniai duomenys
            Console.WriteLine("Testiniai duomenys įkraunami į atmintį \n");
            double[][] X = {
        new double[] { 4,5,7,10 }, new double[] { 7,4,2,9 },
        new double[] { 0,6,12,15 }, new double[] { 1,4,8,10 },
        new double[] { 9,7,5,0 }, new double[] { 14,7,0,-1 },
        new double[] { 6,9,12,-20 }, new double[] { 8,9,10,14 },
        new double[] { 1.45,0.43,-12,15 }, new double[] { 8.13, -9, 0.22,-0.5 },
        new double[] { 5,7.5,9,13 }, new double[] { 2.45, -8, -0.22,-0.5 }};

            int[] y = { -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1 };  

            for (int i = 0; i < X.Length; ++i)
            {
                Console.Write(y[i].ToString().PadLeft(4) + " | ");
                for (int j = 0; j < X[i].Length; ++j)
                {
                    Console.Write(X[i][j].ToString("F1").PadLeft(6));
                }
                Console.WriteLine("");
            }

            // generuojamas ir apmokomas SVM 
            Console.WriteLine("\nSukuriamas ir apmokomas Polinominis branduolys SVM");
            var smo = new SequentialMinimalOptimization<Polynomial>();
            smo.Complexity = 1.0;
            smo.Kernel = new Polynomial(2, 0.0);
            smo.Epsilon = 1.0e-6;
            smo.Tolerance = 1.0e-2;
            Console.WriteLine("\nPradedamas apmokymas");
            var svm = smo.Learn(X, y);
            Console.WriteLine("Apmokymų pabaiga");

            // tikrinamas SVM modelis
            Console.WriteLine("\nTikrinamas SVM modelio teisingumas \n");
            bool[] preds = svm.Decide(X);
            double[] score = svm.Score(X);  // sprendimo funkcija

            int numCorrect = 0; int numWrong = 0;
            for (int i = 0; i < preds.Length; ++i)
            {
                Console.Write("Prognozuota reiksme (double) : " +
                  score[i].ToString("F4").PadLeft(8) + "    ");
                Console.Write("Prognozuota reiksme  (bool): " +
                  preds[i].ToString().PadLeft(6) + "   ");
                Console.WriteLine("Tikra reiksme: " + y[i].ToString().PadLeft(3));
                if (preds[i] == true && y[i] == 1)
                    ++numCorrect;
                else if (preds[i] == false && y[i] == -1)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            double acc = (numCorrect * 100.0) / (numCorrect + numWrong);
            Console.WriteLine("\nModelio tikslumas = " +
              acc.ToString("F2") + "%");

            Console.WriteLine("Iveskite pirma reiksme");
            double First = Convert.ToDouble(Console.ReadLine());
            
            Console.WriteLine("Iveskite antra reiksme");
            double Second = Convert.ToDouble(Console.ReadLine());

            Console.WriteLine("Iveskite trecia reiksme");
            double Third = Convert.ToDouble(Console.ReadLine()); ;

            Console.WriteLine("Iveskite ketvirta reiksme");
            double Fourth = Convert.ToDouble(Console.ReadLine());

            // modelio naudojimas
            bool predClass = svm.Decide(new double[] { First, Second, Third, Fourth });
            Console.WriteLine("\nPrognozuota klase ivestos reiksmems [{0}, {1}, {2}, {3}] = {4}", First, Second, Third, Fourth, predClass);

            // modelio informacija
            Console.WriteLine("\nModelio atraminiai vektoriai: ");
            double[][] sVectors = svm.SupportVectors;
            for (int i = 0; i < sVectors.Length; ++i)
            {
                for (int j = 0; j < sVectors[i].Length; ++j)
                {
                    Console.Write(sVectors[i][j].ToString("F1") + "   ");
                }
                Console.WriteLine("");
            }

            Console.WriteLine("\nModelio svoriai: ");
            double[] wts = svm.Weights;
            for (int i = 0; i < wts.Length; ++i)
                Console.Write(wts[i].ToString("F6") + " ");
            Console.WriteLine("");

            double b = svm.Threshold;
            Console.WriteLine("\nModelio b poslinkis = " + b.ToString("F6"));

            Console.WriteLine("\nSVM demo pabaiga ");
            Console.ReadLine();
        } 
    } 
}


