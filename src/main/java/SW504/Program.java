package SW504;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
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
    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    //Number of possible labels = number of outputs
    protected static int numLabels = 5;
    //Batchsize is the number of pictures loaded in memory at runtime
    protected static int batchSize = 32;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int iterations = 1;
    protected static int epochs = 10;
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

    }

    private static void UiServerSetup(){
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        //trainedNetwork.setListeners(new StatsListener(statsStorage));
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
                .modelSaver(new LocalFileModelSaver("trained_model.zip"))
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
        trainedNetwork.setListeners(new ScoreIterationListener(1));
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
                .addLayer("conv1", convInit("conv1", channels, 32, new int[]{3, 3}, new int[]{2, 2}, new int[]{0, 0}, 0), "input")
                .addLayer("conv2", convInit("conv2", 32, 32, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0), "conv1")
                .addLayer("conv3Padded", convInit("conv3Padded", 32, 64, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0), "conv2")
                .addLayer("pool1", maxPool("pool1", new int[]{3, 3}), "conv3Padded")
                .addLayer("conv4", convInit("conv4", 64, 80, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0), "pool1")
                .addLayer("conv5", convInit("conv5", 80, 192, new int[]{3, 3}, new int[]{2, 2}, new int[]{0, 0}, 0), "conv4")
                .addLayer("conv6", convInit("conv6", 192, 288, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0), "conv5");
        nextInput = inception5Builder(graph, "inception1", 3, "conv6");
        nextInput = inception6Builder(graph, "inception2", 5, nextInput);
        nextInput = inception7Builder(graph, "inception3", 2, nextInput);

        graph
                .addLayer("pool2", maxPool("pool2",new int[]{8,8}),nextInput)
                .addLayer("logic", new DenseLayer.Builder().nOut(1000).nIn(2048).build(),"pool2")
                .setOutputs("classifier")
                .addLayer("classifier", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nOut(numLabels).nIn(250).build(),"logic");


        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        return model;

    }

    public static String inception5Builder(ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale, String input) {
        int blockInput = 288;
        int out16 = 16, out32 = 32,out96 = 96,out128 = 128,out64 = 64;

        String previousBlock = input;
        for(int i=1; i<=scale; i++) {
            graph
                    // 1x1 -> 3x3 -> 3x3
                    .addLayer(nameLayer(blockName,"inception1Conv1",i),convInit("inception1Conv1",blockInput,out16,new int[]{1,1},new int[]{1,1},new int[]{0,0},0),previousBlock)
                    .addLayer(nameLayer(blockName,"inception1Conv2",i),convInit("inceptionConv2",out16,out32,new int[]{3,3},new int[]{1,1},new int[]{0,0},0),nameLayer(blockName,"inception1Conv1",i))
                    .addLayer(nameLayer(blockName,"inception1Conv3",i),convInit("inceptionConv3",out32,out32,new int[]{3,3},new int[]{1,1},new int[]{0,0},0),nameLayer(blockName,"inception1Conv2",i))
                    //1x1 -> 3x3
                    .addLayer(nameLayer(blockName,"inception1Conv4",i),convInit("inception1Conv4",blockInput,out96,new int[]{1,1},new int[]{1,1},new int[]{0,0},0),previousBlock)
                    .addLayer(nameLayer(blockName,"inception1Conv5",i),convInit("inceptionConv5",out96,out128,new int[]{3,3},new int[]{1,1},new int[]{0,0},0),nameLayer(blockName,"inception1Conv4",i))
                    //pool -> 1x1
                    .addLayer(nameLayer(blockName,"pool1",i),maxPool("pool1",new int[]{3,3}),previousBlock)
                    .addLayer(nameLayer(blockName,"inception1Conv6",i),convInit("inceptionConv6",blockInput,out32,new int[]{3,3},new int[]{1,1},new int[]{0,0},0),nameLayer(blockName,"pool1",i))
                    //1x1
                    .addLayer(nameLayer(blockName,"inception1Conv7",i),convInit("inceptionConv7",blockInput,out64,new int[]{1,1},new int[]{1,1},new int[]{0,0},0),previousBlock)
                    //Merge
                    .addVertex(nameLayer(blockName,"merge1",i),new MergeVertex(),nameLayer(blockName,"inception1Conv3",i),nameLayer(blockName,"inception1Conv5",i),nameLayer(blockName,"inception1Conv6",i),nameLayer(blockName,"inception1Conv7",i));

            previousBlock = nameLayer(blockName,"merge1",i);
            blockInput = out32 + out128 + out64 + out32;
            out16 += 16;
            out32 += 32;
            out64 += 64;
            out96 += 96;
            out128 += 128;
        }
        return previousBlock;
    }

    public static String inception6Builder(ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale, String input) {
        String previousBlock = input;
        int n = 7;
        for(int i=1; i<=scale; i++) {
            graph
                    // 1x1
                    .addLayer(nameLayer(blockName,"cnn1",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                    // pool -> 1x1
                    .addLayer(nameLayer(blockName,"pool2",i),maxPool("pool2",new int[]{3,3}),previousBlock)
                    .addLayer(nameLayer(blockName,"cnn2",i),convInit("cnn2",256,256,new int[]{1,1},new int[]{1,1},new int[]{0,0},0),nameLayer(blockName,"pool2",i))
                    // 1x1 -> 1xn -> nx1
                    .addLayer(nameLayer(blockName,"cnn3",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                    .addLayer(nameLayer(blockName,"cnn4",i), new ConvolutionLayer.Builder(new int[]{1, n}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn3",i))
                    .addLayer(nameLayer(blockName,"cnn5",i), new ConvolutionLayer.Builder(new int[]{n, 1}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn4",i))
                    // 1x1 -> 1xn -> nx1 -> 1xn -> nx1
                    .addLayer(nameLayer(blockName,"cnn6",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                    .addLayer(nameLayer(blockName,"cnn7",i), new ConvolutionLayer.Builder(new int[]{1, n}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn6",i))
                    .addLayer(nameLayer(blockName,"cnn8",i), new ConvolutionLayer.Builder(new int[]{n, 1}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn7",i))
                    .addLayer(nameLayer(blockName,"cnn9",i), new ConvolutionLayer.Builder(new int[]{1, n}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn8",i))
                    .addLayer(nameLayer(blockName,"cnn10",i), new ConvolutionLayer.Builder(new int[]{n, 1}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn9",i))
                    // merge
                    .addVertex(nameLayer(blockName,"merge2",i),new MergeVertex(),nameLayer(blockName,"cnn2",i),nameLayer(blockName,"cnn5",i),nameLayer(blockName,"cnn9",i));


            previousBlock = nameLayer(blockName,"merge2",i);
        }
        return previousBlock;
    }

    public static String inception7Builder(ComputationGraphConfiguration.GraphBuilder graph, String blockName, int scale, String input) {
        String previousBlock = input;
        for(int i=1; i<=scale; i++) {
            graph
                    // 1x1
                    .addLayer(nameLayer(blockName,"cnn1",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                    // pool -> 1x1
                    .addLayer(nameLayer(blockName,"pool2",i),maxPool("pool2",new int[]{3,3}),previousBlock)
                    .addLayer(nameLayer(blockName,"cnn2",i),convInit("cnn2",256,256,new int[]{1,1},new int[]{1,1},new int[]{0,0},0),nameLayer(blockName,"pool2",i))
                    // 1x1 -> 1x3 ; 3x1
                    .addLayer(nameLayer(blockName,"cnn4",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                    .addLayer(nameLayer(blockName,"cnn5",i), new ConvolutionLayer.Builder(new int[]{1, 3}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn4",i))
                    .addLayer(nameLayer(blockName,"cnn6",i), new ConvolutionLayer.Builder(new int[]{3, 1}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn4",i))
                    // 1x1 -> 3x3 -> 1x3 ; 3x1
                    .addLayer(nameLayer(blockName,"cnn7",i), new ConvolutionLayer.Builder(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(32).build(), previousBlock)
                    .addLayer(nameLayer(blockName,"cnn8",i), new ConvolutionLayer.Builder(new int[]{3, 3}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn7",i))
                    .addLayer(nameLayer(blockName,"cnn9",i), new ConvolutionLayer.Builder(new int[]{3, 1}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn8",i))
                    .addLayer(nameLayer(blockName,"cnn10",i), new ConvolutionLayer.Builder(new int[]{1, 3}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(32).build(), nameLayer(blockName,"cnn8",i))
                    // Merge
                    .addVertex(nameLayer(blockName,"merge3",i),new MergeVertex(),nameLayer(blockName,"cnn1",i),nameLayer(blockName,"cnn2",i),nameLayer(blockName,"cnn6",i),nameLayer(blockName,"cnn10",i));

            previousBlock = nameLayer(blockName,"merge3",i);
        }
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
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(0.9))
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.001)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
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
