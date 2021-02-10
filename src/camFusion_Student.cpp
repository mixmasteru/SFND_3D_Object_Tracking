
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor,
                         cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx,
                         cv::Mat &RT) {
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto &lidarPoint : lidarPoints) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (auto it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(lidarPoint);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto &boundingBoxe : boundingBoxes) {
        // create randomized color for current 3D object
        cv::RNG rng(boundingBoxe.boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        double xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto &lidarPoint : boundingBoxe.lidarPoints) {
            // world coordinates
            double xw = lidarPoint.x; // world position in m with x facing forward from sensor
            double yw = lidarPoint.y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", boundingBoxe.boxID, (int) boundingBoxe.lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait) {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {
    std::vector<double> eucliDist;
    double eucliDistMean = 0;
    for (auto &kptMatch : kptMatches) {
        const auto &currKeyPoint = kptsCurr[kptMatch.trainIdx];
        if (boundingBox.roi.contains(currKeyPoint.pt)) {
            eucliDist.push_back(cv::norm(currKeyPoint.pt - kptsPrev[kptMatch.queryIdx].pt));
        }
    }
    if(eucliDist.size() != 0){
        eucliDistMean = std::accumulate(eucliDist.begin(), eucliDist.end(), 0.0) / eucliDist.size();
    }

    for (auto &kptMatch : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr[kptMatch.trainIdx].pt)) {
            double dist = cv::norm(kptsCurr[kptMatch.trainIdx].pt - kptsPrev[kptMatch.queryIdx].pt);
            if (dist < eucliDistMean) {
                boundingBox.keypoints.push_back(kptsCurr[kptMatch.trainIdx]);
                boundingBox.kptMatches.push_back(kptMatch);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      const std::vector<cv::DMatch>& kptMatches,
                      double frameRate,
                      double &TTC, cv::Mat *visImg) {

    double dT = 1 / frameRate;
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto &kptMatch1 : kptMatches) { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(kptMatch1.trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(kptMatch1.queryIdx);

        for (auto &kptMatch2 : kptMatches) { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(kptMatch2.trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(kptMatch2.queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }// eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.empty()) {
        TTC = NAN;
        return;
    }

    if (distRatios.size() % 2 == 0) {
        const auto median_it1 = distRatios.begin() + distRatios.size() / 2 - 1;
        const auto median_it2 = distRatios.begin() + distRatios.size() / 2;

        std::nth_element(distRatios.begin(), median_it1, distRatios.end());
        const auto e1 = *median_it1;

        std::nth_element(distRatios.begin(), median_it2, distRatios.end());
        const auto e2 = *median_it2;

        auto median = (e1 + e2) / 2;
        TTC = -dT / (1 - median);

    } else {
        const auto median_it = distRatios.begin() + distRatios.size() / 2;
        std::nth_element(distRatios.begin(), median_it, distRatios.end());
        auto median = *median_it;
        TTC = -dT / (1 - median);
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
    // auxiliary variables
    double dT = 1 / frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto &it : lidarPointsPrev) {

        if (abs(it.y) <= laneWidth / 2.0) { // 3D point within ego lane?
            minXPrev = minXPrev > it.x ? it.x : minXPrev;
        }
    }

    for (auto &it : lidarPointsCurr) {
        if (abs(it.y) <= laneWidth / 2.0) { // 3D point within ego lane?
            minXCurr = minXCurr > it.x ? it.x : minXCurr;
        }
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
    for (auto &boundingBox : prevFrame.boundingBoxes) {
        std::vector<cv::DMatch> boundingBoxMatches;
        std::multimap<int, int> trainIdsInBox;
        int max_cnt = 0;
        int boxId1 = -1;

        for (auto &match : matches) {
            if (boundingBox.roi.contains(prevFrame.keypoints.at(match.queryIdx).pt)) {
                boundingBoxMatches.push_back(match);
            }
        }

        for (auto &boundingBoxMatch : boundingBoxMatches) {
            for (auto &boundingBox1 : currFrame.boundingBoxes) {
                if (boundingBox1.roi.contains(currFrame.keypoints.at(boundingBoxMatch.trainIdx).pt)) {
                    trainIdsInBox.insert(std::pair<int, int>(boundingBox1.boxID, boundingBoxMatch.trainIdx));
                }
            }
        }

        if (!trainIdsInBox.empty()) {
            for (auto &trainId : trainIdsInBox) {
                if (trainIdsInBox.count(trainId.first) > max_cnt) {
                    max_cnt = trainIdsInBox.count(trainId.first);
                    boxId1 = trainId.first;
                }
            }
            if (boxId1 > 0) {
                bbBestMatches.insert(std::pair<int, int>(boundingBox.boxID, boxId1));
            }
        }
    }
}