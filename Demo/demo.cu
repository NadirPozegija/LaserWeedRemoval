/*
 ============================================================================
 Name        : test.cu
 Author      : nadir
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <ctime> //clock to time the various functions of the code
#include <stdio.h>
#include <numeric>
#include <stdlib.h>
//For linker purposes the libraries are found in /usr/lib
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudacodec.hpp>

using namespace cv;
using namespace std;


//------------------------------------------------Global Variables-----------------------------------------------------------------------

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
const Mat z = Mat::zeros(Size(1080, 1080), CV_8UC1); //1080x1080 is the size of the images that will be processed
const string gst = "gst-launch-1.0 v4l2src device=\"/dev/video0\" ! video/x-raw,format=UYVY, width=1920,height=1080 ! xvimagesink";
vector<Mat> images;
float percentThresholdWeedDetection = 0.006; //1 %
int totalPixels = 1166400; //1080x1080=1166400 pixels
vector<Mat> cells;
int cellSideLength = 270; //will create 4x4 tiles from image with dimensions 270x270
int rows, cols = 1080;

//Clock variables
clock_t time_sec;
clock_t time_total_sec;

//------------------------------------------------End of Global Variables------------------------------------------------------------

//------------------------------------------------Function Declarations------------------------------------------------//

Mat IsolateCrCbChannels(Mat &CrCb, Mat &image); //return a cv::Mat and pass in the image or camera frame
vector<Mat> ReadImages(void);
bool IsWeedDetected( const Mat &mask );
bool IsWeedDetected( const vector<Mat> &cells );
void CreateCells(Mat &mask);

//---------------------------------------------End of Function Declarations--------------------------------------------//



#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */


__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = 1.0/data[idx];
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	return rc;
}

float *cpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	for (unsigned cnt = 0; cnt < size; ++cnt) rc[cnt] = 1.0/data[cnt];
	return rc;
}


void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5*(i+1);
}

int main(void)
{
	//Preloaded Stuff
	static const int WORK_SIZE = 65530;
	float *data = new float[WORK_SIZE];

	initialize (data, WORK_SIZE);

	float *recCpu = cpuReciprocal(data, WORK_SIZE);
	float *recGpu = gpuReciprocal(data, WORK_SIZE);
	float cpuSum = std::accumulate (recCpu, recCpu+WORK_SIZE, 0.0);
	float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);

	/* Verify the results */
	//std::cout<<"gpuSum = "<<gpuSum<< " cpuSum = " <<cpuSum<<std::endl;

	/* Free memory */
	delete[] data;
	delete[] recCpu;
	delete[] recGpu;
	//End of Preloaded Stuff



//***************Pretend the camera is streaming properly**********************


//	*****************GSTREAMER or default camera?*******************
	//cap has two ways to go, either cap(0) and let it be default
    //or set the gst pipeline
	system("echo iwillremember | sudo -S /home/ubuntu/Pre_Production_Testing/camera_initialize.sh"); //Camera Set up
	//printf("Camera Set Up \n");
	//cv::VideoCapture cap("/home/ubuntu/Videos/demoVids/garage.mp4");
	cv::VideoCapture cap(0);
	if (!cap.isOpened()){
		printf("Camera NOT opened! \n");
		return -1;
	}
	cap.set(CV_CAP_PROP_FRAME_WIDTH,1920);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
	cout << "Width x Height = " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << " x " << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << "\n";


//	*****************default camera is very corrupted, GSTREAMER Pipeline does not grab image data*******************
    // View video

    cv::Mat frame;

	Mat yuv_im_indirect, resultYCB;
	Mat CrCb, maskYCB, yuv_im, current_im;

	vector<Mat> YUVchannels;

	//Threshold values
	int min = 40; //Came to this after checking values of multiple plants
	int max = 125; //and images and doing the conversion from RGB

	//Thresholds for masking
	Scalar minYCB = Scalar(0, min, min); //check constructor for cv::Scalar
	Scalar maxYCB = Scalar(0, max, max); //the range for the CrCb channels is 40-125

	Mat fullFrame;

    while(true) {
    	//system("echo iwillremember | sudo -S /home/ubuntu/Pre_Production_Testing/still_image.sh"); //take pic
    	//printf("Pic Taken");
    	//cv::VideoCapture cap("/home/ubuntu/Pre_Production_Testing/CurrentFrame.png");

    	cap >> fullFrame;
    	//waitKey(1);
   		//imshow("vid", fullFrame);  DELETE THESE
    	//waitKey(15);

    	if(fullFrame.empty()){
    		break;
    	}

        cap >> frame;  // Get a new frame from camera
    	frame = frame(Rect(0,0,1080,1080));


    	cvtColor(frame, yuv_im, COLOR_BGR2YCrCb);  //BGR conversion

    	/*
    	cv::Size s = yuv_im.size();
    	int Srows = s.height;
    	int Scols = s.width;
    	cout << "Width x Height = " << Scols << " x " << Srows << "\n";
    	 */

		CrCb = IsolateCrCbChannels(CrCb, yuv_im);

		inRange(CrCb, minYCB, maxYCB, maskYCB); //create the binary mask

        // Display output video
      	imshow("Display window", frame);
      	if(waitKey(1) >= 0) break;
    }


/*
     ***************Pretend the camera is streaming properly**********************


    //system("/home/ubuntu/Pre_Production_Testing/disp.sh"); //works
    //for(;;){
    	//system("/home/ubuntu/Pre_Production_Testing/still_image.sh");
    	cv::Mat dst;
    	cv::Mat im = imread("/home/ubuntu/Pre_Production_Testing/test.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
    	imshow("Display Window", im);
    	waitKey(1);
    	waitKey(150);
    	system("/home/ubuntu/Pre_Production_Testing/clear.sh");
    	cv::destroyAllWindows();

    //}

	//cuda::cvtColor(frame, dst, COLOR_YUV2RGB_Y422);

	//********************************************************************************
*/

	//read in images
	//make vector of images to be used in the demo
	//images = ReadImages();


	//7 images
	//loop through them all and convert to YCrCb
	//Then mask them and display the result
	//Then work on the "detection"
	//When to start firing the laser?

	//cuda::GpuMat gpu_yuv_im, gpu_current_im;
    /* ALREADY DECLARED
	Mat yuv_im_indirect, resultYCB;
	Mat CrCb, maskYCB, yuv_im, current_im;

	vector<Mat> YUVchannels;

	//Threshold values
	int min = 40; //Came to this after checking values of multiple plants
	int max = 125; //and images and doing the conversion from RGB

	//Thresholds for masking
	Scalar minYCB = Scalar(0, min, min); //check constructor for cv::Scalar
	Scalar maxYCB = Scalar(0, max, max); //the range for the CrCb channels is 40-125


	for(int i = 0; i < 7; ++i){  //********Change back to 7*********
		current_im = (images[i]);

		imshow("RGB image", current_im); waitKey(0);

		cvtColor(current_im, yuv_im, COLOR_BGR2YCrCb);  //BGR -> YUV conversion
		imshow("YUV image", yuv_im); waitKey(0);

		time_sec = clock();
		time_total_sec = clock();
		CrCb = IsolateCrCbChannels(CrCb, yuv_im);
		time_sec = clock() - time_sec;
		cout << "Time spent Isolating CrCb Channels " << (float)time_sec/CLOCKS_PER_SEC << " seconds" << endl;
		imshow("CrCb image", CrCb); waitKey(0);

		time_sec = clock();
		inRange(CrCb, minYCB, maxYCB, maskYCB); //create the binary mask
		time_sec = clock() - time_sec;
		cout << "Time spent creating binary mask " << (float)time_sec/CLOCKS_PER_SEC << " seconds" << endl;


		imshow("mask", maskYCB); waitKey(0);

		//now to create cells
		time_sec = clock();
		CreateCells(maskYCB);
		time_sec = clock() - time_sec;
		cout << "Time spent creating 16 tiles from the image " << (float)time_sec/CLOCKS_PER_SEC << " seconds" << endl;

		bool detect = false;
		time_sec = clock();
		detect = IsWeedDetected(cells);
		time_sec = clock() - time_sec;
		if(detect){
			cout << "Weed detected, turn on lasers" << endl;
			cout << "Time it took to run detection " << (float)time_sec/CLOCKS_PER_SEC << endl;
		}
		else{
			cout << "No weed detected" << endl;
			cout << "Time it took to run detection " << (float)time_sec/CLOCKS_PER_SEC << endl;
		}
		time_total_sec = clock() - time_total_sec;
		cout << "Time to do all of the computation " << (float)time_total_sec/CLOCKS_PER_SEC << endl << endl;

		destroyAllWindows();

	}
	*/

	/*
	 * Simulate a camera stream with a short video
	 */

	/*
	VideoCapture cap("/home/ubuntu/Videos/demoVids/garage.avi", CAP_ANY);
	VideoWriter video("", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(1080, 1080));

	if(!cap.isOpened()){
		cout << "Error opening video stream" << endl;
		return 1;
	}

	Mat fullFrame;
	//int rows, cols = 1080;

	while(1){
		cap >> fullFrame;
		waitKey(1);
		imshow("vid", fullFrame);
		waitKey(15);

		if(fullFrame.empty()){
			break;
		}
		//crop the incoming video stream to 1080x1080
		//Mat frame = Mat(rows, cols);

	}
	*/




	return 0;
}


//------------------------------------------------FUNCTIONS------------------------------------------------//

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

Mat IsolateCrCbChannels(Mat &CrCb, Mat &image){

	vector<Mat> YUVchannels;
	split(image, YUVchannels);
	YUVchannels[0] = z;
	merge(YUVchannels,CrCb);

	return CrCb;
}

vector<Mat> ReadImages(void){

	Mat im_indirect_sun = imread("/home/ubuntu/Pictures/demoimages/cropped_gradient.jpg", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	Mat im_direct_sun = imread("/home/ubuntu/Pictures/transfer/P_A_final_43_270.jpg", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	Mat im_beer_wall = imread("/home/ubuntu/Pictures/demoimages/cropped_beer.jpg", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	Mat im_plant1 = imread("/home/ubuntu/Pictures/demoimages/cropped_plant1.jpg", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	Mat dirt = imread("/home/ubuntu/Pictures/transfer/dirt.jpg", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	Mat im_gradient = imread("/home/ubuntu/Pictures/transfer/P_A_final_23_90.jpg", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	Mat im_test_track = imread("/home/ubuntu/Pictures/demoimages/cropped_test_track.jpg", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	images.push_back(im_indirect_sun); images.push_back(im_direct_sun); images.push_back(im_beer_wall); images.push_back(im_plant1);
	images.push_back(dirt); images.push_back(im_gradient); images.push_back(im_test_track);
	return images;
}

void CreateCells(Mat &mask){
	int height = mask.rows; int width = mask.cols;

	for(int i = 0; i < height; i += cellSideLength){
		for(int j = 0; j < width; j += cellSideLength){

			Rect grid(i,j,cellSideLength,cellSideLength); //create the square grid
			Mat matGrid;
			matGrid = mask(grid); //the subregion
			cells.push_back(matGrid); //pushback the subregions

			//imshow("i, j", matGrid); //0-15 grids
			//waitKey(0);
		}
	}

}

bool IsWeedDetected(const Mat &mask){
	/*
	 * This function will work in the following way:
	 * 1st - find a percentage threshold that will activate the laser
	 * Control - search every pixel in one thread, count the hits, and time it for a control value
	 * 2nd - sample every other pixel in the image and count white pixels. Multiply by 2 to get approx number of hits
	 * 	   - test performance of this, both time and accuracy. Find the speedup
	 * 3rd - Launch thread and start from both (0,0) and (1080,1080). Meet in the middle. Time it and find speedup
	 * 4th - Launch thread and sample every other pixel and determine performance
	 * 5th - if time permits, try morphological closing and see if that helps
	 * 6th - if able, take a video and see how well it does with a video stream
	 *
	 *Nothing beats the countNonZeros() in terms of computation time
	 *
	 * Split the image into boxes, count the white pixels in these boxes, if threshold is met in these boxes,
	 * see if they are connected
	 *
	 */

	/*
	 * 1st generation detection algorithm, not good enough
	 */
	int white = countNonZero(mask);
	cout << "White pixels = " << white << endl;
	float percent = (float)white/totalPixels;
	cout << "percent = " << percent << endl;

	if(percent >= percentThresholdWeedDetection){
		return true;
	}
	else{
		return false;
	}
}

bool IsWeedDetected(const vector<Mat> &cells){
	/*
	 * The idea here is a more robust detection scheme. The masked image has been separated into 16 tiles
	 * each with dimension 270x270 pixels. From here we will test each tile individually. In the off chance that maybe
	 * a plant or area of interest was cut off by the tiles we will test adjacent tiles until we have exhausted all
	 * the possible search criteria.
	 * There are 49 total unique areas to search. The areas are as follows:
	 * The first 16 are the just the 16 individual tiles
	 * The next 12 are one base tile and the tile below it. They will combined and searched as one tile
	 * The next 12 are one base tile and the tile to the right of it. They will be combined and searched as one tile
	 * The last 9 are the groups of tiles that can be made by combining 4 tiles into one square tile. This is in case
	 * a plant or weed was split into 4 tiles with the center of the weed being the corner that the tiles all share
	 *
	 * If any tile is saturated enough then the function will immeadiately return true. Otherwise we will count white
	 * pixels in each tile and store those values in a vector. If we get through all 16 tiles we simply use the values
	 * in the vector and dont have to recount anything just do simple arithmetic
	 */

	/*
	 * The thresholding values in this are still being figured out but the core functionality works and is solid
	 *
	 * My reasoning for the threshold value I chose. If the crop row is 40 inches across and the image fully
	 * captures the entire crop row from end-to-end and I split the image into a 4x4 grid then each grid is
	 * about 10x10 inches in real life. With a window of 10"x10" a typical weed that we would want to detect is maybe
	 * 3 or 4 inches by 3 or 4 inches. so minimum is 81x81 pixels of green plant plus a little bit of noise from the
	 * dirt and background.
	 */



	vector<int> whitePixelsVec;
	int whiteVal;
	int thresholdOneTile = 6500;    // this is about ~9% of the tile being white.
	int thresholdTwoTiles = 7000;   // the extra 500 is to account for noise
	int thresholdFourTiles = 8000;  // The extra 1000 is to account for noise again

	for(int i = 0; i < 15; i++){
		whiteVal = countNonZero(cells[i]);
		whitePixelsVec.push_back(whiteVal);
		if(whiteVal > thresholdOneTile){ //if the number of white pixels in the current tile is > threshold
			cout << "1 tile case detected at tile " << i << endl;
			return true;    			 //  tile is > threshold then return true and you are done
		}
	}

	//If the code gets here then none of the tiles had enough to warrant the lasers turning on.
	//now we check the tile combinations.
	//first is 2 tiles horizontally. there are 12 combinations

	for(int i = 0; i <= 11; i++){
		if( (whitePixelsVec[i] + whitePixelsVec[i+4]) > thresholdTwoTiles){ //this will add a tile to the one to the right
			cout << "2 tile case detected at tile " << i << " and " << i+4 << endl;
			return true;
		}
	}

	//Check the other 12 cases with two tiles. A tile plus the tile below it
	for(int i = 0; i <= 14; i++){
		if(i == 3 || i == 7 || i == 11){
			i++; //this will skip the rightmost tiles and prevent any weird wraparound
				 //wraparound is bad becuase that has no physical meaning
		}
		if( (whitePixelsVec[i] + whitePixelsVec[i+1]) > thresholdTwoTiles){ //add tile to the tile below
			cout << "2 tile case detected at tile " << i << " and " << i+1 << endl;
			return true;
		}
	}

	//check the last case, 4 tiles together
	for(int i = 0; i <= 10; ++i){
		if(i == 2 || i == 6){
			i++; //again to prevent wraparound
		}
		if( (whitePixelsVec[i] + whitePixelsVec[i+1] + whitePixelsVec[i+4] + whitePixelsVec[i+5]) > thresholdFourTiles){
			cout << "4 tile case detected at tile " << i << endl;
			return true;
		}
	}

	//if it makes it down here then nothing was detected
	return false;

}

//--------------------------------------------------END----------------------------------------------------//
