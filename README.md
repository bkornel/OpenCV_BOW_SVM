# OpenCV_BOW_SVM

A simple object classifier with Bag-of-Words using OpenCV 3.0

The vocabulary is created by [`BOWKMeansTrainer`][1] and we have a feature detector, extractor, matcher and a BOW image descriptor extractor (to compute an image descriptor using the bag of visual words) such as:

	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SURF");
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("SURF");
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce ");

	cv::BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide->setVocabulary(vocabulary);

First of all we need to scour the training set for our histograms:
	
	cv::Mat samples;
	cv::Mat labels(0, 1, CV_32FC1);
		
	for(auto& it : imagePosDir)
	{
		cv::Mat image = cv::imread(it);
		
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(image, keypoints);
		
		if(keypoints.empty()) continue;
		
		// Responses to the vocabulary
		cv::Mat imgDescriptor;
		bowide.compute(image, keypoints, imgDescriptor);
		
		if(imgDescriptor.empty()) continue;
		
		if(samples.empty())
		{
			samples.create(0, imgDescriptor.cols, imgDescriptor.type());
		}
		
		// Copy class samples and labels
		std::cout << "Adding " << imgDescriptor.rows << " positive sample." << std::endl;
		samples.push_back(imgDescriptor);
		
		cv::Mat classLabels = cv::Mat::ones(imgDescriptor.rows, 1, CV_32FC1);
		labels.push_back(classLabels);
	}

Do the same for `imagePosNeg` except that `classLabels` will have zero values, such as:

	...
	cv::Mat classLabels = cv::Mat::zeros(imgDescriptor.rows, 1, CV_32FC1);
    labels.push_back(classLabels);
	...

Note how I build the samples and the labels, I marked the positive samples with labels '1', and then the negatives with label '0'. So we have the training data for each class (here for positives and negatives) in `samples`. Lets's get training:

	cv::Mat samples_32f; 
	samples.convertTo(samples_32f, CV_32F);
	
	CvSVM svm; 
	svm.train(samples_32f, labels);
	// Do something with the classifier, like saving it to file
	
Then testing let's get testing the classifier:

	for(auto& it : testDir)
	{
		cv::Mat image = cv::imread(it);
		
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(image, keypoints);
		
		if(keypoints.empty()) continue;
		
		// Responses to the vocabulary
		cv::Mat imgDescriptor;
		bowide.compute(image, keypoints, imgDescriptor);
		
		if(imgDescriptor.empty()) continue;
		
		float res = svm.predict(imgDescriptor, true);
		
		std::cout << "- Result of prediction: " << res << std::endl;
	}
	

  [1]: http://docs.opencv.org/modules/features2d/doc/object_categorization.html#bowkmeanstrainer
