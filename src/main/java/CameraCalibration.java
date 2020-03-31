import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class CameraCalibration {
    Mat cameraMatrix;
    MatOfDouble distCoeffs;

    public static List<Mat> frames= new ArrayList<>();

    public CameraCalibration(){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        cameraMatrix = new Mat();
        distCoeffs =  new MatOfDouble();


        File rootDir= new File("chessboard");
        File[] files = rootDir.listFiles();

        for(File file :files) {
            String src = file.getAbsolutePath();
            Mat imgread = Imgcodecs.imread(src);
            frames.add(imgread);

            calibrateCamera(7, 7, frames);
        }
    }


    public void calibrateCamera(int pNumInnerCornersX, int pNumInnerCornersY, List<Mat> pTrainingImages) {
        int board_w = pNumInnerCornersX;
        int board_h = pNumInnerCornersY;
        Size board_sz = new Size(board_w, board_h);
        //Real location of the corners in 3D
        MatOfPoint2f corners = new MatOfPoint2f();
        Size imageSize = null;
        List<Mat>imagePoints = new ArrayList<Mat>();

        List<Mat> objs = new ArrayList<Mat>();
        for(Mat img : pTrainingImages) {
            Mat gray = new Mat();
            imageSize = img.size();

            int board_n = board_w*board_h;

            MatOfPoint3f obj = new MatOfPoint3f();
            for (int j=0; j<board_n; j++)
            {
                obj.push_back(new MatOfPoint3f(new Point3((double)j/(double)board_w, (double)j%(double)board_w, 0.0d)));

            }
            objs.add(obj);


            Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
            boolean findChessboardCorners = Calib3d.findChessboardCorners(gray, board_sz,corners);
            if(findChessboardCorners) {
                TermCriteria aCriteria = new TermCriteria(TermCriteria.EPS |
                        TermCriteria.MAX_ITER, 30,0.1);
                Imgproc.cornerSubPix(gray, corners, new Size(11,11), new Size(-1,-1), aCriteria);
                Calib3d.drawChessboardCorners(gray, board_sz, corners, true);
                imagePoints.add(corners);
            }

        }
        performCalibration(objs, imagePoints, imageSize);
    }



    private double performCalibration(List<Mat> pObj, List<Mat> pImagePoints, Size pImageSize) {
        cameraMatrix = new Mat(3,3,CvType.CV_32FC1);
        distCoeffs = new MatOfDouble();
        List<Mat> rvecs = new ArrayList<Mat>();
        List<Mat> tvecs = new ArrayList<Mat>();
        Mat perViewError = new Mat();
        double error = Calib3d.calibrateCameraExtended(pObj, pImagePoints, pImageSize, cameraMatrix, distCoeffs, rvecs, tvecs, new Mat(), new Mat(), perViewError);
        scaleCameraMatrix(640, 480, 1080, 720, cameraMatrix);
        return error;
    }

    public Mat getCameraMatrix() {
        return cameraMatrix;
    }

    /*Since changing the resolution of the camera does not affect the distortion coefficients, but only affects the camera matrix. That being said, when the resolution is changed,
    *all values in the camera matrix are scaled proportionally to the change in resolution, hence, we can auto scale the camera matrix so you don't have to recalibrate.
    *@param  oldDimX   this is the old resolution along the x axis
    *@param  oldDimY   this is the old resolution along the y axis
    *@param  newDimX   this is the new resolution along the x axis
    *@param  newDimY   this is the new resolution along the y axis
    */
    public void scaleCameraMatrix(double oldDimX, double oldDimY, double newDimX, double newDimY, Mat cameraMatrix){
        //The focal length and center of image along the x axis
        double fx =  cameraMatrix.get(0, 0)[0];
        double cx = cameraMatrix.get(0, 2)[0];

        //The focal length and center of image along the y axis
        double fy =  cameraMatrix.get(1, 1)[0];
        double cy = cameraMatrix.get(1, 2)[0];

        //Replace fx, fy, cx,, and cy in the Mat with the new scaled ones
        cameraMatrix.put(0, 0, fx * (newDimX / oldDimX));
        cameraMatrix.put(1, 1, fy * (newDimY / oldDimY));

        cameraMatrix.put(0, 2, cx * (newDimX / oldDimX));
        cameraMatrix.put(1, 2, cy * (newDimY / oldDimY));

    }

    public MatOfDouble getDistCoeffs() {
        return distCoeffs;
    }
}


