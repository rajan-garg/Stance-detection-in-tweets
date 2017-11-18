from feature_extraction import FeatureExtraction
from text_helper import TextHelper

from config import DIC_FILE, AFF_FILE, dummy_stop_words


if __name__ == '__main__':

	textHelper = TextHelper(DIC_FILE, AFF_FILE, dummy_stop_words)

	featureExtraction = FeatureExtraction(textHelper)
	
	featureExtraction.run()
