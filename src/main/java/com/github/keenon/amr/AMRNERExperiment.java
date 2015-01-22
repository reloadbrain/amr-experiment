package com.github.keenon.amr;

import com.github.keenon.minimalml.engine.Engine;
import com.github.keenon.minimalml.engine.MultiThreadedEngine;
import com.github.keenon.minimalml.engine.SingleThreadedEngine;
import com.github.keenon.minimalml.experiment.Experiment;
import com.github.keenon.minimalml.kernels.Kernel;
import com.github.keenon.minimalml.kernels.LinearKernel;
import com.github.keenon.minimalml.optimizer.Optimizer;
import com.github.keenon.minimalml.optimizer.QNOptimizer;
import com.github.keenon.minimalml.optimizer.SGDOptimizer;
import com.github.keenon.minimalml.reader.DataReader;
import com.github.keenon.minimalml.reader.LabeledSequenceDataReader;
import com.github.keenon.minimalml.utils.TransferMap;
import com.github.keenon.minimalml.word2vec.Word2VecLoader;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;

import java.io.IOException;

/**
 * Created by keenon on 1/8/15.
 *
 * Implements an experiment class
 */
public class AMRNERExperiment extends Experiment {
    TransferMap<String, double[]> embeddings;

    public AMRNERExperiment() {
        try {
            embeddings = Word2VecLoader.loadDataAsynch("data/google-300-trimmed.ser.gz");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Kernel getKernel() {
        return new LinearKernel();
    }

    @Override
    public Engine getEngine() {
        return new SingleThreadedEngine();
    }

    @Override
    public String getTrainingSetPath() {
        return "data/train-400-seq.txt";
    }

    @Override
    public String getTestSetPath() {
        return "data/test-100-seq.txt";
    }

    @Override
    public String getReportPath() {
        return "data/amr-ner-experiment-report";
    }

    @Override
    public Optimizer getOptimizer() {
        // regularization
        return new QNOptimizer();
    }

    @Override
    public DataReader getReader() {
        return new LabeledSequenceDataReader(
                new LabeledSequenceDataReader.LabeledSequenceFeature[]{
                        // Target feature
                        new LabeledSequenceDataReader.LabeledSequenceStringFeature() {
                            @Override
                            public String getName() {
                                return "AMR Action";
                            }

                            @Override
                            public String featurize(int offset,
                                                    String[] tokens,
                                                    String[] labels,
                                                    Annotation annotation) {
                                return labels[offset];
                            }
                        },

                        // Token feature
                        new LabeledSequenceDataReader.LabeledSequenceStringFeature() {
                            @Override
                            public String getName() {
                                return "Token string";
                            }

                            @Override
                            public String featurize(int offset,
                                                    String[] tokens,
                                                    String[] labels,
                                                    Annotation annotation) {
                                return tokens[offset];
                            }
                        },

                        // Word embedding
                        new LabeledSequenceDataReader.LabeledSequenceDoubleArrayFeature() {
                            @Override
                            public String getName() {
                                return "Google 300-dim word embeddings";
                            }

                            @Override
                            public double[] featurize(int offset,
                                                     String[] tokens,
                                                     String[] labels,
                                                     Annotation annotation) {
                                double[] f = embeddings.getBlocking(tokens[offset]);
                                if (f == null) {
                                    // empty, inner product will be 0, won't bias decisions at all
                                    return new double[300];
                                }
                                return f;
                            }
                        },

                        // NER Tags
                        new LabeledSequenceDataReader.LabeledSequenceStringFeature() {
                            @Override
                            public String getName() {
                                return "NER Tags";
                            }

                            @Override
                            public String featurize(int offset,
                                                     String[] tokens,
                                                     String[] labels,
                                                     Annotation annotation) {
                                return annotation.get(CoreAnnotations.TokensAnnotation.class).get(offset)
                                        .get(CoreAnnotations.NamedEntityTagAnnotation.class);
                            }
                        }
                }
        );
    }

    @Override
    public int getTargetFeature() {
        return 0;
    }

    @Override
    public AblativeAnalysisType getAblativeAnalysisType() {
        return AblativeAnalysisType.LAST_FEATURE;
    }

    public static void main(String[] args) throws Exception {
        Experiment exp = new AMRNERExperiment();
        exp.run();
    }
}

