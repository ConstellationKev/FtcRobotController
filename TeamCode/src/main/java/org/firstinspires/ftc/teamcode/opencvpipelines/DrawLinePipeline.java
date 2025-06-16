package org.firstinspires.ftc.teamcode.opencvpipelines;

import java.sql.Array;
import java.util.*;

import org.opencv.core.Core;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

public class DrawLinePipeline extends OpenCvPipeline {
    Point prevPoint;
    Point newPoint;
    ArrayList<Point> points = new ArrayList<>();

    Mat output = new Mat();

    Mat kernal = Imgproc.getStructuringElement(0, new Size(5, 5));

    public static double distance(double x1, double y1, double x2, double y2) {
        return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }

    public Mat drawLine(Mat imageFrame) {
        double contourArea = 6000;
        double distanceThreshold = 30;
        Mat hsvFrame = new Mat();
        Mat redMask = new Mat();
        Mat redRes = new Mat();
        ArrayList<MatOfPoint> redContours = new ArrayList<>();
        double pointx;
        double pointy;

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
                 pointx = (x.tl().x + x.br().x)/2;
                 pointy = (x.tl().y + x.br().y)/2;
                 newPoint = new Point(pointx, pointy);
                if (prevPoint == null) {
                    prevPoint = new Point(pointx, pointy);
                } else {
                    if (distance(newPoint.x, newPoint.y, prevPoint.x, prevPoint.y) > distanceThreshold) {
                        points.add(newPoint);
                        prevPoint = new Point(pointx, pointy);
                    }
                }
            }
        }

        if (points.size()>1) {
            for (int i = 0; i<points.size()-1; i++) {
                Imgproc.line(imageFrame, points.get(i), points.get(i+1), new Scalar(255, 255, 255), 2);
            }
        }

        return imageFrame;
    }

    @Override
    public Mat processFrame(Mat input)
    {
        output = drawLine(input);
        return output;
    }
}
