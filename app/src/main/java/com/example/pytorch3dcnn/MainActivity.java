package com.example.pytorch3dcnn;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.MediaController;
import android.widget.VideoView;

import org.pytorch.IValue;
//import org.pytorch.MemoryFormat;
import org.pytorch.Module;
//import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;

import static org.pytorch.IValue.listFrom;

//import org.opencv.core.Mat;
//import static org.opencv.core.CvType.CV_32F;



public class MainActivity extends AppCompatActivity {

    private final String TAG="PyTorch ver.1";

    //public Module model=null;
    private File videoFile;
    private Uri videoFileUri;
    private MediaMetadataRetriever retriever;
    public ArrayList<Bitmap> bitmapArrayList;
    private MediaPlayer mediaPlayer;
    public Bitmap bitmap;
    private Thread thread;

    VideoView myVideo;
    private MediaController media_control;

    public ArrayList<Tensor> tensorArrayList;
    //public Tensor[] TensorList;
    public ArrayList<byte[]> byteArrayList;
    public byte[] bb;
    public long[] shape;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);

        setContentView(R.layout.activity_main);


        myVideo = (VideoView) findViewById(R.id.videoView);
        //Uri uri = Uri.parse("android.resource://" + getPackageName() + "/raw/video");
        // 혹은 R.raw.video

        bitmapArrayList = getFrames("android.resource://" + getPackageName() + "/raw/video");
        //System.out.println(bitmapArrayList.size());

        Tensor[] TensorList = new Tensor[160];
        tensorArrayList = new ArrayList<Tensor>();
        byteArrayList = new ArrayList<byte[]>();
        //final IValue = IValue.from

        for(int i=0; i<bitmapArrayList.size(); i++){
            bitmap = bitmapArrayList.get(i);
            //bb = bitmapToByteArray(bitmap);

            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, 0, 0, 128, 128, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

            TensorList[i] = inputTensor;

            //System.out.println(TensorList[i]);
            //System.out.println(inputTensor);

            //IValue.listFrom(inputTensor).toTensorList();
            //tensorArrayList.add(inputTensor);

            //byteArrayList.add(bb);
            //shape = new long[]{1, bb.length};
            //Tensor inputTensor = Tensor.fromBlobUnsigned(bb, shape);
        }
        //IValue.listFrom((IValue)TensorList).toTensorList();


        System.out.println(TensorList.length);
        //System.out.println(tensorArrayList);
        //System.out.println(tensorArrayList.get(0));

        //Byte[] bytes = (Byte[]) tensorArrayList.toArray();
        //final long[] shape = new long[]{1, bb.length};
        //final Tensor inputTensor = Tensor.fromBlobUnsigned(bytes, shape);

        //IValue[] tt = (IValue.tupleFrom(IValue.listFrom(TensorList))).toTuple();
        Tensor[] tt = (IValue.listFrom(TensorList)).toTensorList();

        //FloatBuffer buffer = Tensor.allocateFloatBuffer(1*3*160*128*128);
        //Tensor tt = Tensor.fromBlob(buffer, new long[]{1, 3, 160, 224, 224});


        try {
            Module convert = Module.load(assetFilePath(this, "convert.pt"));
            Tensor inputTensor = convert.runMethod("list_to_tensor", IValue.listFrom(tt)).toTensor();
            System.out.println(inputTensor);
            //System.out.println(inputTensor.getDataAsFloatArray());
            System.out.println("입력 Tensor?:"+IValue.from(inputTensor).isTensor());
            System.out.println("입력 NULL?:"+IValue.from(inputTensor).isNull());


            Module model = Module.load(assetFilePath(this, "PhysNet_v2.pkl"));
            //System.out.println("앞:"+IValue.from(inputTensor));

            System.out.println("출력 NULL?:"+ model.forward(IValue.from(inputTensor)).isNull());
            Tensor outputTensor = model.runMethod("forward", IValue.from(inputTensor)).toTensor();
            //String outputTensor = model.forward(IValue.from(inputTensor)).toString();
            //System.out.println("뒤:"+IValue.from(inputTensor));

            System.out.println(IValue.from(inputTensor).isTensor());
            System.out.println("출력:"+outputTensor);


            //Tensor inputTensor = model.runMethod("list_to_tensor", IValue.listFrom(tt)).toTensor();
            //Tensor[] outputTensor = model.forward(IValue.listFrom(tt)).toTensorList();
            //Tensor outputTensor = model.runMethod("sumTensorsList", IValue.listFrom(TensorList)).toTensor();
            //float[] result = outputTensor.getDataAsFloatArray();
        } catch (IOException e) {
            e.printStackTrace();
        }


        //Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);


        //myVideo.setVideoURI(uri);
        //myVideo.start();


        /*
        try {
            //Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));

            model=Module.load(assetFilePath(this, "PhysNet.pkl"));

            //InputStream mat = getAssets().open("image.mat");
            //DataInputStream input = new DataInputStream(mat);

            //Bitmap bitmap = new Bitmap("C://Users//SM-PC//Desktop//PyTorch3dcnn//app//src//main//assets//image.mat");

            //double d = input.readDouble();
            //System.out.println(d);
            //File mat = new File("C://Users//SM-PC//Desktop//PyTorch3dcnn//app//src//main//assets//image.mat");

            //byte[] res = inputStreamToByteArray(mat);

            //float[] fArray = new float[0];
            //Random randomGenerator = new Random();
            //float randomDouble = randomGenerator.nextFloat();

            //System.out.println(fArray.length);
            //byte[] bArray= new byte[res];
            //System.arraycopy(fArray,0,res,0,res.length);
            //System.out.println(res.length);
            //Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mat, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
            //FloatBuffer buffer = Tensor.allocateFloatBuffer(1*3*160*128*128);
            //FloatBuffer buffer = FloatBuffer.allocate(1*3*160*128*128);
            //Tensor inputTensor = Tensor.fromBlob(fArray, new long[]{1*3*160*128*128});
            //System.out.println(inputTensor);
            //ArrayList<Bitmap> bitmap = getFrames("C://Users//SM-PC//Desktop//PyTorch3dcnn//app//src//main//assets//video.mp4");
            //System.out.println(bitmap);


            //File videoFile = new File("C:/Users/SM-PC/Desktop/PyTorch3dcnn/app/src/main/assets/video.mp4");
            //Uri uri =  Uri.parse(videoFile.toString());

            //System.out.println(uri);
            //String videoPath = "C://Users//SM-PC//Desktop//PyTorch3dcnn//app//src//main//assets//video.mp4";
            //System.out.println(videoPath);

            //File file = new File("C://Users//SM-PC//Desktop//PyTorch3dcnn//app//src//main//assets//video.mp4");
            //Intent intent = new Intent(Intent.ACTION_VIEW);
            //intent.setDataAndType(Uri.fromFile(file), "video/*");
            //startActivity(intent);




            videoFileUri = Uri.parse("android.resource://" + getPackageName() + "/" + R.raw.video);

            retriever = new MediaMetadataRetriever();
            bitmapArrayList = new ArrayList<Bitmap>();
            retriever.setDataSource(videoFile.toString());


            mediaPlayer = MediaPlayer.create(getBaseContext(), videoFileUri);
            int millisecond = mediaPlayer.getDuration();
            for(int i = 0; i < millisecond; i+=1000){
                bitmap = retriever.getFrameAtTime(i*1000,MediaMetadataRetriever.OPTION_CLOSEST);
                bitmapArrayList.add(bitmap);
            }
            retriever.release();

            thread = new Thread(new Runnable(){
                @Override
                public void run() {
                    try {
                        saveFrames(bitmapArrayList);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });
            thread.start();


            //mediaMetadataRetriever.setDataSource(uri);
            //Bitmap bitmap = mediaMetadataRetriever.getFrameAtTime(160000000);//160초 영상 추출
            //System.out.println(bitmap);

            //Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

            /*long size = 0;
            int chunk = 0;
            long start = 0;
            InputStream is = null;
            //File largeFile = new File("C://Users//SM-PC//Desktop//PyTorch3dcnn//app//src//main//assets//image.mat");
            InputStream largeFile = getAssets().open("image.mat");
            try {
                start = System.currentTimeMillis();
                is = new FileInputStream(largeFile);
                byte[] buffer = new byte[1024];
                while((chunk = is.read(buffer)) != -1){
                    size += chunk;
                }
            } catch (FileNotFoundException e) {
                System.out.println("Failed to open file stream:" + e.getMessage());
            } catch (IOException e) {
                System.out.println("Failed to read from file stream:" + e.getMessage());
            }finally{
                if(is != null){
                    try {
                        is.close();
                    } catch (IOException e) {
                        System.out.println("Failed to close InputStream: " + e.getMessage());
                    }
                }
                long done = System.currentTimeMillis() - start;
                System.out.println(String.format("took %d milliseconds", done));
            }
            System.out.println("size: " + size + " bytes");
            */

            //Mat floatMat = new Mat();
            //mat.convertTo(floatMat, CV_32F);
            //FloatBuffer floatBuffer = floatMat.createBuffer();
            //Tensor.create(new long[]{1, image_height, image_width, 3}, floatBuffer);

            //Tensor outputTensor = model.forward(IValue.from(tensorArrayList)).toTensor();
            //float[] result = outputTensor.getDataAsFloatArray();
            //System.out.println(Arrays.toString(result));
        //} catch (IOException e) {
        //    e.printStackTrace();
        //    System.out.println("에러1");
        //}

    }

    public void test(Tensor inputTensor) throws IOException {
        //final Module model=Module.load(assetFilePath(this, "PhysNet.pkl"));
        //final IValue input = IValue.from(Tensor.fromBlob(Tensor.allocateByteBuffer(1), new long[] {1}));

        final IValue input = IValue.from(inputTensor);
        //assertTrue(input.isTensor());
        //final IValue output = model.forward(input);
        //assertTrue(output.isNull());
        //System.out.println(input);
        input.toTensor();
        System.out.println(input);
        //test2(input);
    }

    public void test2(IValue inputTensor) throws IOException {
        final Module model=Module.load(assetFilePath(this, "PhysNet.pkl"));
        //final IValue input = IValue.from(Tensor.fromBlob(Tensor.allocateByteBuffer(1), new long[] {1}));

        inputTensor.toTensorList();
        //assertTrue(input.isTensor());
        final IValue output = model.forward(IValue.listFrom(inputTensor));
        //assertTrue(output.isNull());
        System.out.println(output);
    }

    public void saveFrames(ArrayList<Bitmap> saveBitmap) throws IOException{
        String folder = Environment.getExternalStorageDirectory().toString();
        File saveFolder = new File(folder + "//");
        if(!saveFolder.exists()){
            saveFolder.mkdirs();
        }
        int i = 1;
        for (Bitmap b : saveBitmap){
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            b.compress(Bitmap.CompressFormat.JPEG, 40, bytes);
            File f = new File(saveFolder,("filename"+i+".jpg"));

            f.createNewFile();
            FileOutputStream fo = new FileOutputStream(f);
            fo.write(bytes.toByteArray());

            fo.flush();
            fo.close();
            i++;
        }
        thread.interrupt();
    }


    public String getRealPathFromURI(Uri contentUri) {

        String[] proj = { MediaStore.Images.Media.DATA };

        Cursor cursor = getContentResolver().query(contentUri, proj, null, null, null);
        cursor.moveToNext();
        String path = cursor.getString(cursor.getColumnIndex(MediaStore.MediaColumns.DATA));
        Uri uri = Uri.fromFile(new File(path));

        Log.d(TAG, "getRealPathFromURI(), path : " + uri.toString());

        cursor.close();
        return path;
    }

    public byte[] bitmapToByteArray (Bitmap bitmap){
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        return byteArray;
    }

    private ArrayList<Bitmap> getFrames(String path) {
        try {
            ArrayList<Bitmap> bArray = new ArrayList<Bitmap>();
            bArray.clear();
            MediaMetadataRetriever mRetriever = new MediaMetadataRetriever();
            mRetriever.setDataSource(this, Uri.parse(path));
            //mRetriever.getFrameAtTime(160000000);
            //int millisecond = mediaPlayer.getDuration();
            for(int i=1000000;i<=160000000;i+=1000000)
            {
                bArray.add(mRetriever.getFrameAtTime(i,
                        MediaMetadataRetriever.OPTION_CLOSEST_SYNC));
            }
            return bArray;
        } catch (Exception e) { return null;}
    }

    public static byte[] inputStreamToByteArray(InputStream is) {

        byte[] resBytes = null;
        ByteArrayOutputStream bos = new ByteArrayOutputStream();

        byte[] buffer = new byte[4];
        int read = -1;
        try {
            while ( (read = is.read(buffer)) != -1 ) {
                bos.write(buffer, 0, read);
            }

            resBytes = bos.toByteArray();
            bos.close();
        }
        catch (IOException e) {
            e.printStackTrace();
        }

        return resBytes;
    }


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file=new File(context.getFilesDir(), assetName);
        if(file.exists() && file.length()>0){
            return file.getAbsolutePath();
        }

        try(InputStream is=context.getAssets().open(assetName)){
            try(OutputStream os=new FileOutputStream(file)){
                byte[] buffer=new byte[4]; // 에러나면 가장 의심되는 부분
                int read;
                while((read=is.read(buffer))!=-1){
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}