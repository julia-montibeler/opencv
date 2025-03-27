#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;

using namespace cv;

int main() {

     // Open the default camera (0)
     cv::VideoCapture cap(0);

     // Check if the camera is opened successfully
     if (!cap.isOpened()) {
         std::cout << "Error: Could not open video capture." << std::endl;
         return -1;
     }
 
     // Declare a frame variable to store the captured frame
    cv::Mat frame, fundo, img;
    
    while (true) {
        cap >> fundo;

        cv::Mat grayImage;
        cv::cvtColor(fundo, grayImage, COLOR_BGR2GRAY);
    
        cv::Mat blurredImage;
        cv::GaussianBlur(grayImage, blurredImage, Size(5, 5), 1.5);
        cv::imshow("Edge Detection", grayImage);
    
        cv::Mat edges;
        int lowThreshold = 50;
        int highThreshold = 150;
        cv::Canny(blurredImage, edges, lowThreshold, highThreshold);
    
        // Display the original and the edge-detected image
        // cv::imshow("Edge Detection", edges);

        if (cv::waitKey(1) == 27) {
            break;  // Exit the loop if 'Esc' is pressed
        }


    }
     


    // Close all OpenCV windows
    cv::destroyAllWindows();

    return 0;
}