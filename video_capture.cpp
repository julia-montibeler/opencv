#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

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
    cv::Mat frame;

    // Capture and display video until 'Esc' key is pressed
    while (true) {
        // Capture a new frame from the camera
        cap >> frame;

        // Check if the frame is empty (in case of an error)
        if (frame.empty()) {
            std::cout << "Error: Failed to capture frame." << std::endl;
            break;
        }

        // Show the captured frame in a window
        cv::imshow("Video Capture", frame);

        // Wait for 1 ms and check if the 'Esc' key (ASCII 27) is pressed
        if (cv::waitKey(1) == 27) {
            break;  // Exit the loop if 'Esc' is pressed
        }

        Vec3b pixel = frame.at<Vec3b>(100, 100);

        // O valor do pixel em cada canal: B, G e R
        std::cout << "B = " << (int)pixel[0] << ", G = " << (int)pixel[1] << ", R = " << (int)pixel[2] << std::endl;
    }

    // Release the camera and close all OpenCV windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
