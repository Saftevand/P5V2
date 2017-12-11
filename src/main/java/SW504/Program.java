package SW504;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.zoo.model.AlexNet;
import org.deeplearning4j.zoo.model.InceptionResNetV1;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.InceptionResNetV1;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Program {
    //Height Width and channels of pictures
    protected static int height = 99;
    protected static int width = 99;
    protected static int channels = 3;
    //Number of possible labels = number of outputs
    protected static int numLabels = 5;
    //Batchsize is the number of pictures loaded in memory at runtime
    protected static int batchSize = 16;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int iterations = 1;
    protected static int epochs = 15;
    //Paths for test and training sets
    protected static File trainPath  =  new File("C:/Users/palmi/Desktop/NeuralTesting/trainSet");
    protected static File testPath = new File("C:/Users/palmi/Desktop/NeuralTesting/testSet");
    private static FileSplit trainData = new FileSplit(trainPath, NativeImageLoader.ALLOWED_FORMATS,rng);
    private static FileSplit testData = new FileSplit(testPath,NativeImageLoader.ALLOWED_FORMATS,rng);
    protected static boolean trainWithTransform = false;
    protected static MultiLayerNetwork trainedNetwork = alexnetModel();
    private static Logger log = LoggerFactory.getLogger(Program.class);
    private static ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
    private static ImageRecordReader trainingReader = new ImageRecordReader(height,width,channels,labelGenerator);
    private static ImageRecordReader testReader = new ImageRecordReader(height,width,channels,labelGenerator);


    public static void main(String[] args) throws Exception {
        initImageReaders();

        trainedNetwork = normalTraining(trainedNetwork);
        evaluateNetwork(testReader,new ImagePreProcessingScaler(0,1),testData,trainedNetwork);
        saveNetwork("trained_model.zip",trainedNetwork);

        /*
        ComputationGraph network = inceptionModel();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners(new StatsListener(statsStorage));
        DataSetIterator dataIterator = new RecordReaderDataSetIterator(trainingReader,batchSize,1,numLabels);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIterator);
        dataIterator.setPreProcessor(scaler);
        for( int i=0; i<epochs; i++ ){
            System.out.println("Running Epoch: " + i);
            network.fit(dataIterator);
        }
        */
    }

    private static void UiServerSetup(){
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        trainedNetwork.setListeners(new StatsListener(statsStorage));
    }

    private static void initImageReaders() throws IOException {
        trainingReader.initialize(trainData);
        trainingReader.setListeners(new LogRecordListener());
        testReader.initialize(testData);

    }

    private static MultiLayerNetwork  trainNetwork_withTransformation(ImageRecordReader recordReader, InputSplit trainSet, MultiLayerNetwork neuralNetwork) throws IOException {
        DataSetIterator dataIterator;
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        MultipleEpochsIterator trainIter;
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);

        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});

        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainSet, transform);
            dataIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIterator);
            dataIterator.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIterator);
            neuralNetwork.fit(trainIter);
        }

        return neuralNetwork;
    }

    private static void saveNetwork(String fileName, MultiLayerNetwork neuralNetwork) throws IOException {
        File saveTo = new File(fileName);
        ModelSerializer.writeModel(neuralNetwork,saveTo,false);
    }

    private static void evaluateNetwork(ImageRecordReader recordReader, DataNormalization scaler, InputSplit fileSet, MultiLayerNetwork neuralNetwork) throws IOException {
        recordReader.reset();
        recordReader.initialize(fileSet);
        DataSetIterator testIte = new RecordReaderDataSetIterator(recordReader,batchSize,1,numLabels);
        scaler.fit(testIte);
        testIte.setPreProcessor(scaler);

        Evaluation eval = new Evaluation(numLabels);

        while(testIte.hasNext()){
            DataSet next = testIte.next();
            INDArray output = neuralNetwork.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(),output);
        }
        log.info(eval.stats());
    }

    private static SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private static void earlyStopTraining() throws IOException {

        ImageRecordReader validationReader = new ImageRecordReader(height,width,channels,labelGenerator);
        ImageRecordReader trainingReader = new ImageRecordReader(height,width,channels,labelGenerator);

        validationReader.initialize(testData);
        trainingReader.initialize(trainData);

        DataSetIterator validationSet = new RecordReaderDataSetIterator(validationReader,batchSize,1, numLabels);
        DataSetIterator trainingSet = new RecordReaderDataSetIterator(trainingReader,batchSize,1,numLabels);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(epochs))
                .scoreCalculator(new DataSetLossCalculator(validationSet,true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("dir"))
                .saveLastModel(true)
                .build();

        EarlyStoppingTrainer esTrainer = new EarlyStoppingTrainer(esConf,alexnetConf(),trainingSet);
        EarlyStoppingResult result = esTrainer.fit();
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());
        //Print score vs. epoch
        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<Integer>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Epoch vs. Score:");
        for( Integer i : list){
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }
    }

    private static MultiLayerNetwork normalTraining(MultiLayerNetwork model) throws Exception{
        DataSetIterator dataIterator = new RecordReaderDataSetIterator(trainingReader,batchSize,1,numLabels);
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIterator);
        dataIterator.setPreProcessor(scaler);
        trainedNetwork.init();
        UiServerSetup();
        for(int i = 0; i < epochs; i++){
            System.out.println("Running Epoch: " + i);
            model.fit(dataIterator);
        }

        return model;
    }

    public static ComputationGraph inceptionModel() {

        String nextInput;
        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.8)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.001)
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder();

        graph
                .addInputs("input")
                .addLayer("conv1", convInit("conv1", channels, 32, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0), "input")
                .addLayer("conv2", convInit("conv2", 32, 32, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0), "conv1")
                .addLayer("conv3Padded", convInit("conv3Padded", 32, 64, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0), "conv2")
                //.addLayer("pool1", maxPool("pool1", new int[]{3, 3}), "conv3Padded")
                .addLayer("conv4", convInit("conv4", 64, 80, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0), "conv3Padded")
                .addLayer("conv5", convInit("conv5", 80, 192, new int[]{3, 3}, new int[]{2, 2}, new int[]{0, 0}, 0), "conv4")
                .addLayer("conv6", convInit("conv6", 192, 288, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0), "conv5");

        //3x factorized, regular inception layers
        nextInput = inceptionFive(graph, "incep1", "conv6");
        nextInput = inceptionFive(graph, "incep2", nextInput);
        nextInput = inceptionFive(graph, "incep3", nextInput);

        //Grid reduction
        nextInput = inceptionGridReductionOne(graph, "gridReduc1", nextInput);

        //5x even more factorized regular inception layers
        nextInput = inceptionSix(graph, "incep4", nextInput);
        nextInput = inceptionSix(graph, "incep5", nextInput);
        nextInput = inceptionSix(graph, "incep6", nextInput);
        nextInput = inceptionSix(graph, "incep7", nextInput);
        nextInput = inceptionSix(graph, "incep8", nextInput);

        //Grid reduction version 2, more harder and stronger and better. (Faktisk just the same)
        nextInput = inceptionGridReductionTwo(graph, "gridReduc2", nextInput);

        //2x dank ass inception from figure 7.
        nextInput = inceptionSeven(graph, "incep9", nextInput, new int[][]{{192},{128},{224,256,256},{128,192,224,224}});
        nextInput = inceptionSeven(graph, "incep10", nextInput, new int[][]{{320},{192},{384,384,384},{448,384,384,384}});

        graph
                .addLayer("pool2", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build(),nextInput)
                //.addLayer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{8,8}, new int[]{1,1}, new int[]{0,0}).build(),nextInput)
                .addLayer("logic", new DenseLayer.Builder().nIn(2048).nOut(1000).biasInit(1).build(), "pool2")
                .setOutputs("classifier")
                .addLayer("classifier", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nOut(numLabels).nIn(1000).build(),"logic");
                //.setInputTypes(InputType.convolutional(299,299,3)).build();


        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        return model;

    }

    public static String inceptionFive(ComputationGraphConfiguration.GraphBuilder graph, String blockName, String input) {
        int blockInput = 288;
        int out16 = 16, out32 = 32,out96 = 96,out128 = 128,out64 = 64;

            graph
                    // 1x1 -> 3x3 -> 3x3
                    .addLayer("cnn1"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0, 0).nOut(out16).build(), input)
                    .addLayer("cnn2"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(out16).stride(1,1).padding(1, 1).nOut(out32).build(), "cnn1"+blockName)
                    .addLayer("cnn3"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(out32).stride(1,1).padding(1, 1).nOut(out64).build(), "cnn2"+blockName)

                    //1x1 -> 3x3
                    .addLayer("cnn4"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0, 0).nOut(out96).build(), input)
                    .addLayer("cnn5"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(out96).stride(1,1).padding(1, 1).nOut(out128).build(), "cnn4"+blockName)

                    //pool -> 1x1
                    //This stride is a guess based on the GoogleNet model which is a predecessor to the inceptionV2 model
                    .addLayer("pool1a"+blockName, new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{1,1}).build(),input)
                    .addLayer("cnn6"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(1, 1).nOut(out32).build(), "pool1a"+blockName)

                    //1x1
                    .addLayer("cnn7"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0, 0).nOut(out64).build(),input)

                    //Merge
                    .addVertex("merge1"+blockName,new MergeVertex(),"cnn3"+blockName,"cnn5"+blockName,"cnn6"+blockName,"cnn7"+blockName);

            String previousBlock = "merge1"+blockName;

            return previousBlock;
    }

    public static String inceptionGridReductionOne(ComputationGraphConfiguration.GraphBuilder graph, String blockName, String input) {
        String previousBlock = input;

        graph
                // 1x1 -> 3x3 -> 3x3
                .addLayer("cnn1"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(288).stride(1,1).nOut(128).build(), previousBlock)
                .addLayer("cnn2"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(128).stride(1,1).nOut(192).build(), "cnn1"+blockName)
                .addLayer("cnn3"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(192).stride(2,2).nOut(288).build(), "cnn2"+blockName)

                //1x1 -> 3x3
                .addLayer("cnn4"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(288).stride(1,1).nOut(128).build(), previousBlock)
                .addLayer("cnn5"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(128).stride(2,2).nOut(192).build(),"cnn4"+blockName)

                //pool -> 1x1
                .addLayer("pool1"+blockName,maxPool("pool1"+blockName,new int[]{3,3}),previousBlock)

                //Merge
                .addVertex("merge1"+blockName,new MergeVertex(),"cnn3"+blockName,"cnn5"+blockName,"pool1"+blockName);

        previousBlock = "merge1"+blockName;
        return previousBlock;
    }

    public static String inceptionSix(ComputationGraphConfiguration.GraphBuilder graph, String blockName, String input) {
        int blockInput = 768;
        int out32 = 32, out64 = 64, out96 = 96, out128 = 128, out160 = 160, out192 = 192, out256 = 256;
        int n = 7;

            graph
                    // 1x1
                    .addLayer("cnn1"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0,0).nOut(out256).build(), input)
                    // pool -> 1x1
                    .addLayer("pool1"+blockName, new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1}).build(),input)
                    .addLayer("cnn2"+blockName,convInit("cnn2"+blockName,blockInput,out96,new int[]{1,1},new int[]{1,1},new int[]{1,1},0),"pool1"+blockName)
                    // 1x1 -> 1xn -> nx1
                    .addLayer("cnn3"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0,0).nOut(out64).build(), input)
                    .addLayer("cnn4"+blockName, new ConvolutionLayer.Builder(new int[]{1, n}).convolutionMode(ConvolutionMode.Same).nIn(out64).stride(1,1).padding(0,3).nOut(out128).build(), "cnn3"+blockName)
                    .addLayer("cnn5"+blockName, new ConvolutionLayer.Builder(new int[]{n, 1}).convolutionMode(ConvolutionMode.Same).nIn(out128).stride(1,1).padding(3,0).nOut(out160).build(), "cnn4"+blockName)
                    // 1x1 -> 1xn -> nx1 -> 1xn -> nx1
                    .addLayer("cnn6"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0,0).nOut(out32).build(), input)
                    .addLayer("cnn7"+blockName, new ConvolutionLayer.Builder(new int[]{1, n}).convolutionMode(ConvolutionMode.Same).nIn(out32).stride(1,1).padding(0,3).nOut(out96).build(), "cnn6"+blockName)
                    .addLayer("cnn8"+blockName, new ConvolutionLayer.Builder(new int[]{n, 1}).convolutionMode(ConvolutionMode.Same).nIn(out96).stride(1,1).padding(3,0).nOut(out160).build(),"cnn7"+blockName)
                    .addLayer("cnn9"+blockName, new ConvolutionLayer.Builder(new int[]{1, n}).convolutionMode(ConvolutionMode.Same).nIn(out160).stride(1,1).padding(0,3).nOut(out192).build(), "cnn8"+blockName)
                    .addLayer("cnn10"+blockName, new ConvolutionLayer.Builder(new int[]{n, 1}).convolutionMode(ConvolutionMode.Same).nIn(out192).stride(1,1).padding(3,0).nOut(out256).build(), "cnn9"+blockName)
                    // merge
                    .addVertex("merge1"+blockName,new MergeVertex(),"cnn1"+blockName,"cnn2"+blockName,"cnn5"+blockName,"cnn10"+blockName);

            String previousBlock = "merge1"+blockName;

        return previousBlock;
    }

    public static String inceptionGridReductionTwo(ComputationGraphConfiguration.GraphBuilder graph, String blockName, String input) {
        String previousBlock = input;
        int blockInput = 768;

        graph
                // 1x1 -> 3x3 -> 3x3
                .addLayer("cnn1"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).nOut(256).build(), previousBlock)
                .addLayer("cnn2"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(256).stride(1,1).nOut(288).build(), "cnn1"+blockName)
                .addLayer("cnn3"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(288).stride(2,2).nOut(288).build(), "cnn2"+blockName)

                //1x1 -> 3x3
                .addLayer("cnn4"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).nOut(192).build(), previousBlock)
                .addLayer("cnn5"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(192).stride(2,2).nOut(224).build(),"cnn4"+blockName)

                //pool -> 1x1
                .addLayer("pool1"+blockName,maxPool("pool1"+blockName,new int[]{3,3}),previousBlock)

                //Merge
                .addVertex("merge1"+blockName,new MergeVertex(),"cnn3"+blockName,"cnn5"+blockName,"pool1"+blockName);

        previousBlock = "merge1"+blockName;
        return previousBlock;
    }

    public static String inceptionSeven(ComputationGraphConfiguration.GraphBuilder graph, String blockName, String input, int[][] array) {
        int blockInput = 1280;

        graph
                // 1x1
                .addLayer("cnn1"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0,0).nOut(array[0][0]).build(), input)
                // pool -> 1x1
                .addLayer("pool1"+blockName, new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1}).build(),input)
                .addLayer("cnn2"+blockName,convInit("cnn2"+blockName,blockInput,array[1][0],new int[]{1,1},new int[]{1,1},new int[]{0,0},0),"pool1"+blockName)
                // 1x1 -> 1x3 ; 3x1
                .addLayer("cnn4"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0,0).nOut(array[2][0]).build(), input)
                .addLayer("cnn5"+blockName, new ConvolutionLayer.Builder(new int[]{1, 3}).convolutionMode(ConvolutionMode.Same).nIn(array[2][0]).stride(1,1).padding(0,1).nOut(array[2][1]).build(), "cnn4"+blockName)
                .addLayer("cnn6"+blockName, new ConvolutionLayer.Builder(new int[]{3, 1}).convolutionMode(ConvolutionMode.Same).nIn(array[2][0]).stride(1,1).padding(1,0).nOut(array[2][2]).build(), "cnn4"+blockName)
                .addVertex("merge1"+blockName,new MergeVertex(),"cnn5"+blockName,"cnn6"+blockName)

                // 1x1 -> 3x3 -> 1x3 ; 3x1
                .addLayer("cnn7"+blockName, new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(blockInput).stride(1,1).padding(0,0).nOut(array[3][0]).build(), input)
                .addLayer("cnn8"+blockName, new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(array[3][0]).stride(1,1).padding(1,1).nOut(array[3][1]).build(), "cnn7"+blockName)
                .addLayer("cnn9"+blockName, new ConvolutionLayer.Builder(new int[]{3, 1}).convolutionMode(ConvolutionMode.Same).nIn(array[3][1]).stride(1,1).padding(1,0).nOut(array[3][2]).build(), "cnn8"+blockName)
                .addLayer("cnn10"+blockName, new ConvolutionLayer.Builder(new int[]{1, 3}).convolutionMode(ConvolutionMode.Same).nIn(array[3][1]).stride(1,1).padding(0,1).nOut(array[3][3]).build(), "cnn8"+blockName)
                .addVertex("merge2"+blockName,new MergeVertex(),"cnn9"+blockName,"cnn10"+blockName)

                // Merge
                .addVertex("merge3"+blockName,new MergeVertex(),"cnn1"+blockName,"cnn2"+blockName,"merge1"+blockName,"merge2"+blockName);

        String previousBlock = "merge3"+blockName;
        return previousBlock;
    }



    public static String nameLayer(String blockName, String layerName, int i) { return blockName+"-"+layerName+"-"+i; }

    public static MultiLayerConfiguration lenetConf(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.0045) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        return conf;
    }

    public static MultiLayerConfiguration alexnetConf(){
        double nonZeroBias = 1;
        double dropOut = 0;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(0.9))
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{2, 2}, new int[]{10, 10}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{3,3}))
                .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3,3}))
                .layer(6,conv3x3("cnn3", 384, 0))
                .layer(7,conv3x3("cnn4", 384, nonZeroBias))
                .layer(8,conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3,3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        return conf;
    }

    public static MultiLayerNetwork lenetModel() {
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.0045) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public static MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = alexnetConf();

        return new MultiLayerNetwork(conf);

    }

    private static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

}
