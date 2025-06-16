package org.firstinspires.ftc.teamcode.opencvpipelines;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.*;

import org.firstinspires.ftc.robotcore.external.Func;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.core.Core;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

public class ObjectTrackingPipline extends OpenCvPipeline {
    Mat output = new Mat();

    Mat kernal = Imgproc.getStructuringElement(0, new Size(5, 5));

    public Mat drawColors(Mat imageFrame) {
        double contourArea = 6000;
        Mat hsvFrame = new Mat();
        Mat redMask = new Mat();
        Mat blueMask = new Mat();
        Mat yellowMask = new Mat();
        Mat redRes = new Mat();
        Mat blueRes = new Mat();
        Mat yellowRes = new Mat();
        ArrayList<MatOfPoint> redContours = new ArrayList<>();
        ArrayList<MatOfPoint> blueContours = new ArrayList<>();
        ArrayList<MatOfPoint> yellowContours = new ArrayList<>();

        Imgproc.cvtColor(imageFrame, hsvFrame, Imgproc.COLOR_RGB2HSV);

        // detecting red objects
        Scalar lowRedHSV = new Scalar(136, 87, 111);
        Scalar highRedHSV = new Scalar(180, 255, 255);
        Core.inRange(hsvFrame, lowRedHSV, highRedHSV, redMask);
        // processes mask to optimize color detection
        Imgproc.dilate(redMask, redMask, kernal);
        Core.bitwise_and(imageFrame, imageFrame, redRes, redMask);
        // finding contours, boxes, and labels
        Imgproc.findContours(redMask, redContours, new Mat(redMask.size(), redMask.type()), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for (MatOfPoint contour : redContours) {
            double area = Imgproc.contourArea(contour);
            if (area>contourArea) {
                Rect x = Imgproc.boundingRect(contour);
                Imgproc.rectangle(imageFrame, x.tl(), x.br(), new Scalar(255,0,0), 2);
                Imgproc.putText(imageFrame, "Red", x.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255,255,255),2);
            }
        }

        // detecting blue objects
        Scalar lowBlueHSV = new Scalar(94, 80, 2);
        Scalar highBlueHSV = new Scalar(100, 171, 255); // TODO: TUNE THIS
        Core.inRange(hsvFrame, lowBlueHSV, highBlueHSV, blueMask);
        // processes mask to optimize color detection
        Imgproc.dilate(blueMask, blueMask, kernal);
        Core.bitwise_and(imageFrame, imageFrame, blueRes, blueMask);
        // finding contours, boxes, and labels
        Imgproc.findContours(blueMask, blueContours, new Mat(blueMask.size(), blueMask.type()), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for (MatOfPoint contour : blueContours) {
            double area = Imgproc.contourArea(contour);
            if (area>contourArea) {
                Rect x = Imgproc.boundingRect(contour);
                Imgproc.rectangle(imageFrame, x.tl(), x.br(), new Scalar(0,0,255), 2);
                Imgproc.putText(imageFrame, "Blue", x.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255,255,255),2);
            }
        }

        // detecting Yellow objects
        Scalar lowYellowHSV = new Scalar(94, 80, 2);
        Scalar highYellowHSV = new Scalar(100, 171, 255); // TODO: TUNE THIS
        Core.inRange(hsvFrame, lowYellowHSV, highYellowHSV, yellowMask);
        // processes mask to optimize color detection
        Imgproc.dilate(yellowMask, yellowMask, kernal);
        Core.bitwise_and(imageFrame, imageFrame, yellowRes, yellowMask);
        // finding contours, boxes, and labels
        Imgproc.findContours(yellowMask, yellowContours, new Mat(yellowMask.size(), yellowMask.type()), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for (MatOfPoint contour : yellowContours) {
            double area = Imgproc.contourArea(contour);
            if (area>contourArea) {
                Rect x = Imgproc.boundingRect(contour);
                Imgproc.rectangle(imageFrame, x.tl(), x.br(), new Scalar(255,255,0), 2);
                Imgproc.putText(imageFrame, "Yellow", x.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255,255,255),2);
            }
        }

        return imageFrame;
    }

    @Override
    public Mat processFrame(Mat input)
    {
        output = drawColors(input);
        return output;
    }
}
