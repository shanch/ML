using Accord.Controls;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.Bayes;
using System.Linq;
using Accord.IO;
using System.Data;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using System;

namespace SampleAccordNETApp
{
    class Program
    {
        static void Main(string[] args)
        {
            
            DataTable table = new ExcelReader("examples.xls").GetWorksheet("Classification - Yin Yang");

            // Convert the DataTable to input and output vectors
            double[][] inputs = table.ToJagged<double>("X", "Y");
            int[] outputs = table.Columns["G"].ToArray<int>();

            TestKernel(inputs, outputs);


            /*int correct = 0;

            for(int i=0; i<inputs.Length; i++)
            {
                double[][] inputsTrain = inputs.RemoveAt(i);
                int[] outputsTrain = outputs.RemoveAt(i);

                double[] inputTest = inputs[i];
                int outputTest = outputs[i];

                if (ValidateKernel(inputsTrain, outputsTrain, inputTest, outputTest)) correct++;

                //break;
            }

            Console.WriteLine((double)correct / inputs.Length);
            */
        }

        static bool ValidateKernel(double[][] inputsTrain, int[] outputsTrain, double[] inputTest, int outputTest)
        {
            // Create a new Sequential Minimal Optimization (SMO) learning 
            // algorithm and estimate the complexity parameter C from data
            var teacher = new SequentialMinimalOptimization<Gaussian>()
            {
                UseComplexityHeuristic = true,
                UseKernelEstimation = true // estimate the kernel from the data
            };

            // Teach the vector machine
            var svm = teacher.Learn(inputsTrain, outputsTrain);

            // Classify the samples using the model
            bool answer = svm.Decide(inputTest);

            // Convert to Int32 so we can plot:
            int zeroOneAnswer = answer ? 1 : -1;

            return outputTest == zeroOneAnswer;
        }

        static void TestKernel(double[][] inputs, int[] outputs)
        {
            // Create a new Sequential Minimal Optimization (SMO) learning 
            // algorithm and estimate the complexity parameter C from data
            var teacher = new SequentialMinimalOptimization<Gaussian>()
            {
                UseComplexityHeuristic = true,
                UseKernelEstimation = true // estimate the kernel from the data
            };

            // Teach the vector machine
            var svm = teacher.Learn(inputs, outputs);

            // Classify the samples using the model
            bool[] answers = svm.Decide(inputs);

            // Convert to Int32 so we can plot:
            int[] zeroOneAnswers = answers.Select(b => b ? 1 : 0).ToArray<int>();

            // Plot the results
            ScatterplotBox.Show("Expected results", inputs, outputs);
            ScatterplotBox.Show("GaussianSVM results", inputs, zeroOneAnswers);
        }


        static void TestLinearSvm(double[][] inputs, int[] outputs)
        {
            // Create a L2-regularized L2-loss optimization algorithm for
            // the dual form of the learning problem. This is *exactly* the
            // same method used by LIBLINEAR when specifying -s 1 in the 
            // command line (i.e. L2R_L2LOSS_SVC_DUAL).
            //
            var teacher = new LinearCoordinateDescent();

            // Teach the vector machine
            var svm = teacher.Learn(inputs, outputs);

            // Classify the samples using the model
            bool[] answers = svm.Decide(inputs);

            // Convert to Int32 so we can plot:
            int[] zeroOneAnswers = answers.Select(b => b ? 1 : 0).ToArray<int>();

            // Plot the results
            ScatterplotBox.Show("Expected results", inputs, outputs);
            ScatterplotBox.Show("LinearSVM results", inputs, zeroOneAnswers);

            // Grab the index of multipliers higher than 0
            int[] idx = teacher.Lagrange.Find(x => x > 0);

            // Select the input vectors for those
            double[][] sv = inputs.Get(idx);

            // Plot the support vectors selected by the machine
            ScatterplotBox.Show("Support vectors", sv).Hold();
        }

        static void TestNaiveBayes(double[][] inputs, int[] outputs)
        {
            outputs = outputs.Select(i => (i == -1) ? 0 : i).ToArray<int>();

            // Create a Naive Bayes learning algorithm
            var teacher = new NaiveBayesLearning<NormalDistribution>();

            // Use the learning algorithm to learn
            var nb = teacher.Learn(inputs, outputs);

            // At this point, the learning algorithm should have
            // figured important details about the problem itself:
            int numberOfClasses = nb.NumberOfClasses; // should be 2 (positive or negative)
            int nunmberOfInputs = nb.NumberOfInputs;  // should be 2 (x and y coordinates)

            // Classify the samples using the model
            int[] answers = nb.Decide(inputs);

            // Plot the results
            ScatterplotBox.Show("Expected results", inputs, outputs);
            ScatterplotBox.Show("Naive Bayes results", inputs, answers)
                .Hold();
        }

        static void TestLogisticRegression(double[][] inputs, int[] outputs)
        {
            // Create iterative re-weighted least squares for logistic regressions
            var teacher = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                MaxIterations = 100,
                Regularization = 1e-6
            };

            // Use the teacher algorithm to learn the regression:
            LogisticRegression lr = teacher.Learn(inputs, outputs);

            // Classify the samples using the model
            bool[] answers = lr.Decide(inputs);

            // Convert to Int32 so we can plot:
            int[] zeroOneAnswers = answers.Select(b => b ? 1: 0).ToArray<int>();

            // Plot the results
            ScatterplotBox.Show("Expected results", inputs, outputs);
            ScatterplotBox.Show("Logistic Regression results", inputs, zeroOneAnswers)
                .Hold();
        }
    }
}
