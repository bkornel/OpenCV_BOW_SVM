#include <map>
#include <utility>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define POSITIVE_LABEL (0)
#define NEGATIVE_LABEL (1)

typedef std::pair<int, cv::Mat> DatabaseElement;
typedef std::vector<DatabaseElement> DatabaseType;

// naming convention: [pos|neg]_[#].jpg
bool loadImages(const std::string& path, DatabaseType& outDatabase)
{
	cv::Mat posImage, negImage;
	int counter = 1;

	std::cout << "- Loading from: " << path << std::endl;

	do 
	{
		posImage = cv::imread(path + "pos_" + std::to_string(counter) + ".jpg");
		negImage = cv::imread(path + "neg_" + std::to_string(counter) + ".jpg");

		if (!posImage.empty())
		{
			outDatabase.push_back(std::make_pair(POSITIVE_LABEL, posImage));
		}

		if (!negImage.empty())
		{
			outDatabase.push_back(std::make_pair(NEGATIVE_LABEL, negImage));
		}

		counter++;
	} 
	while (!(posImage.empty() && negImage.empty()));

	std::cout << "- Number of images loaded: " << outDatabase .size() << std::endl;

	return !outDatabase.empty();
}

bool createVocabulary(const DatabaseType& trainingDb, cv::Mat& outVocabulary)
{
	CV_Assert(!trainingDb.empty());

	cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create();
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SURF::create();

	cv::Mat trainingDescriptors(1, extractor->descriptorSize(), extractor->descriptorType());

	outVocabulary.create(0, 1, CV_32FC1);

	for (auto& it : trainingDb)
	{
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(it.second, keypoints);

		if (!keypoints.empty())
		{
			cv::Mat descriptors;
			extractor->compute(it.second, keypoints, descriptors);

			if (!descriptors.empty())
			{
				std::cout << "- Adding " << descriptors.rows << " training descriptors." << std::endl;
				trainingDescriptors.push_back(descriptors);
			}
			else
			{
				std::cout << "- No descriptors found." << std::endl;
			}
		}
		else
		{
			std::cout << "- No keypoints found." << std::endl;
		}
	}

	if (trainingDescriptors.empty())
	{
		std::cout << "- Training descriptors are empty." << std::endl;
		return false;
	}

	cv::BOWKMeansTrainer bowtrainer(1000);
	bowtrainer.add(trainingDescriptors);
	outVocabulary = bowtrainer.cluster();

	return true;
}

bool scourTrainingSet(const DatabaseType& trainingDb, const cv::Mat& vocabulary, cv::Mat& outSamples, cv::Mat& outLabels)
{
	CV_Assert(!trainingDb.empty());
	CV_Assert(!vocabulary.empty());

	cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create();
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SURF::create();
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

	cv::BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	cv::Mat samples;
	outSamples.create(0, 1, CV_32FC1);
	outLabels.create(0, 1, CV_32SC1);

	for (auto& it : trainingDb)
	{
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(it.second, keypoints);

		if (!keypoints.empty())
		{
			cv::Mat descriptors;
			bowide.compute(it.second, keypoints, descriptors);

			if (!descriptors.empty())
			{
				if (samples.empty())
				{
					samples.create(0, descriptors.cols, descriptors.type());
				}

				// Copy class samples and labels
				std::cout << "- Adding " << descriptors.rows << " positive sample." << std::endl;
				samples.push_back(descriptors);

				cv::Mat classLabels;

				if (it.first == POSITIVE_LABEL)
				{
					classLabels = cv::Mat::zeros(descriptors.rows, 1, CV_32SC1);
				}
				else
				{
					classLabels = cv::Mat::ones(descriptors.rows, 1, CV_32SC1);
				}

				outLabels.push_back(classLabels);
			}
			else
			{
				std::cout << "- No descriptors found." << std::endl;
			}
		}
		else
		{
			std::cout << "- No keypoints found." << std::endl;
		}
	}

	if (samples.empty() || outLabels.empty())
	{
		std::cout << "- Samples are empty." << std::endl;
		return false;
	}

	samples.convertTo(outSamples, CV_32FC1);

	return true;
}

bool trainSVM(const cv::Mat& samples, const cv::Mat& labels, cv::Ptr<cv::ml::SVM>& outSVM)
{
	CV_Assert(!samples.empty() && samples.type() == CV_32FC1);
	CV_Assert(!labels.empty() && labels.type() == CV_32SC1);

	outSVM = cv::ml::SVM::create();

	return outSVM->train(samples, cv::ml::ROW_SAMPLE, labels);
}

bool testSVM(const DatabaseType& trainingDb, const cv::Mat& vocabulary, const cv::Ptr<cv::ml::SVM>& SVM)
{
	CV_Assert(!trainingDb.empty());
	CV_Assert(!vocabulary.empty());

	cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create();
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SURF::create();
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

	cv::BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	for (auto& it : trainingDb)
	{
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(it.second, keypoints);

		if (keypoints.empty()) continue;

		// Responses to the vocabulary
		cv::Mat imgDescriptor;
		bowide.compute(it.second, keypoints, imgDescriptor);

		if (imgDescriptor.empty()) continue;

		cv::Mat results;
		float res = SVM->predict(imgDescriptor);

		std::string predicted_label = (res == POSITIVE_LABEL ? "COCA COLA" : "PEPSI");
		
		std::cout << "- Result of prediction: (" << predicted_label << "): " << res << std::endl;
				
		cv::imshow(predicted_label, it.second);
		cv::waitKey(-1);

		cv::destroyWindow(predicted_label);
	}

	return true;
}

int main()
{
	std::cout << "1. Loading images" << std::endl;

	const std::string& trainingPath = "images/train/";
	const std::string& testingPath = "images/test/";

	DatabaseType trainingDb, testingDb;
	if (!loadImages(trainingPath, trainingDb))
	{
		return -1;
	}

	std::cout << std::endl;

	if (!loadImages(testingPath, testingDb))
	{
		return -1;
	}

	std::cout << std::endl;
	// -------------------------------------------

	std::cout << "2. Creating vocabulary for BOW" << std::endl;

	cv::Mat vocabulary;
	if (!createVocabulary(trainingDb, vocabulary))
	{
		return -1;
	}
	
	std::cout << std::endl;
	// -------------------------------------------

	std::cout << "3. Scour the training set for our histograms" << std::endl;

	cv::Mat samples_32f;
	cv::Mat labels;
	if (!scourTrainingSet(trainingDb, vocabulary, samples_32f, labels))
	{
		return -1;
	}
	
	std::cout << std::endl;
	// -------------------------------------------

	std::cout << "4. Training SVM" << std::endl;

	cv::Ptr<cv::ml::SVM> SVM;
	if (!trainSVM(samples_32f, labels, SVM))
	{
		return -1;
	}

	std::cout << std::endl;
	// -------------------------------------------

	std::cout << "5. Testing SVM" << std::endl;

	if (!testSVM(trainingDb, vocabulary, SVM))
	{
		return -1;
	}

	std::cout << std::endl;

	if (!testSVM(testingDb, vocabulary, SVM))
	{
		return -1;
	}

	std::cout << std::endl;
	// -------------------------------------------

	return 1;
}
