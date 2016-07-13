package drew;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.type.Sentence;
import org.apache.uima.fit.type.Token;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instances;
import org.cleartk.ml.feature.extractor.CleartkExtractor;
import org.cleartk.ml.feature.extractor.CoveredTextExtractor;
import org.cleartk.ml.feature.extractor.FeatureExtractor1;
import org.cleartk.ml.feature.function.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

class ExamplePOSAnnotator extends CleartkSequenceAnnotator<String> {

    private FeatureFunctionExtractor tokenFeatureExtractor;
    private CleartkExtractor contextFeatureExtractor;

    public void initialize(UimaContext context) throws ResourceInitializationException {
        super.initialize(context);

        // a feature extractor that creates features corresponding to the word, the word lower cased
        // the capitalization of the word, the numeric characterization of the word, and character ngram
        // suffixes of length 2 and 3.
        this.tokenFeatureExtractor = new FeatureFunctionExtractor<Token>(
                new CoveredTextExtractor<Token>(),
                new LowerCaseFeatureFunction(),
                new CapitalTypeFeatureFunction(),
                new NumericTypeFeatureFunction(),
                new CharacterNgramFeatureFunction(CharacterNgramFeatureFunction.Orientation.RIGHT_TO_LEFT, 0, 2),
                new CharacterNgramFeatureFunction(CharacterNgramFeatureFunction.Orientation.RIGHT_TO_LEFT, 0, 3));

        // a feature extractor that extracts the surrounding token texts (within the same sentence)
        this.contextFeatureExtractor = new CleartkExtractor<Sentence, Token>(
                Token.class,
                new CoveredTextExtractor<Token>(),
                new CleartkExtractor.Preceding(2),
                new CleartkExtractor.Following(2));
    }


    @Override
    public void process(JCas jCas) throws AnalysisEngineProcessException {
        // for each sentence in the document, generate training/classification instances
        for (Sentence sentence : JCasUtil.select(jCas, Sentence.class)) {
            List<List<Feature>> tokenFeatureLists = new ArrayList<List<Feature>>();
            List<String> tokenOutcomes = new ArrayList<String>();

            // for each token, extract features and the outcome
            List<Token> tokens = JCasUtil.selectCovered(jCas, Token.class, sentence);
            for (Token token : tokens) {

                // apply the two feature extractors
                List<Feature> tokenFeatures = new ArrayList<Feature>();
                tokenFeatures.addAll(this.tokenFeatureExtractor.extract(jCas, token));
                tokenFeatures.addAll(this.contextFeatureExtractor.extractWithin(jCas, token, sentence));
                tokenFeatureLists.add(tokenFeatures);

                // add the expected token label from the part of speech
                if (this.isTraining()) {
                    tokenOutcomes.add(token.getPos());
                }
            }

            // for training, write instances to the data write
            if (this.isTraining()) {
                this.dataWriter.write(Instances.toInstances(tokenOutcomes, tokenFeatureLists));
            }

            // for classification, set the token part of speech tags from the classifier outcomes.
            else {
                List<String> outcomes = this.classifier.classify(tokenFeatureLists);
                Iterator<Token> tokensIter = tokens.iterator();
                for (String outcome : outcomes) {
                    tokensIter.next().setPos(outcome);
                }
            }
        }
    }
}
