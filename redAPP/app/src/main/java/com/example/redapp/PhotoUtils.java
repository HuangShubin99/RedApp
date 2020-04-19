package com.example.redapp;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class PhotoUtils {
    //    start camera
    public static String start_camera(Activity activity, int requestCode) {
        Uri imageUri;
        // save image in cache path
        File outputImage = new File(Environment.getExternalStorageDirectory().getAbsolutePath()
                + "/lite_mobile/", System.currentTimeMillis() + ".jpg");
        Log.d("outputImage", outputImage.getAbsolutePath());
        try {
            if (outputImage.exists()) {
                outputImage.delete();
            }
            File out_path = new File(Environment.getExternalStorageDirectory().getAbsolutePath()
                    + "/lite_mobile/");
            if (!out_path.exists()) {
                out_path.mkdirs();
            }
            outputImage.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (Build.VERSION.SDK_INT >= 24) {
            // compatible with Android 7.0 or over
            imageUri = FileProvider.getUriForFile(activity,
                    "com.yeyupiaoling.testtflite.fileprovider", outputImage);
        } else {
            imageUri = Uri.fromFile(outputImage);
        }
        // set system camera Action
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        // set save photo path
        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        // set photo quality, min is 0, max is 1
        intent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 0);
        activity.startActivityForResult(intent, requestCode);
        // return image absolute path
        return outputImage.getAbsolutePath();
    }

    //    get picture in photo
    public static void use_photo(Activity activity, int requestCode) {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        activity.startActivityForResult(intent, requestCode);
    }

    //    get photo from Uri
    public static String get_path_from_URI(Context context, Uri uri) {
        String result;
        Cursor cursor = context.getContentResolver().query(uri, null, null, null, null);
        if (cursor == null) {
            result = uri.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            result = cursor.getString(idx);
            cursor.close();
        }
        return result;
    }

    //    TensorFlow model,get predict data
    public static ByteBuffer getScaledMatrix(Bitmap bitmap, int[] ddims) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(ddims[0] * ddims[1] * ddims[2] * ddims[3] * 4);
        imgData.order(ByteOrder.nativeOrder());
        // get image pixel
        int[] pixels = new int[ddims[2] * ddims[3]];
        Bitmap bm = Bitmap.createScaledBitmap(bitmap, ddims[2], ddims[3], false);
        bm.getPixels(pixels, 0, bm.getWidth(), 0, 0, ddims[2], ddims[3]);
        int pixel = 0;
        for (int i = 0; i < ddims[2]; ++i) {
            for (int j = 0; j < ddims[3]; ++j) {
                final int val = pixels[pixel++];
                imgData.putFloat((val & 0xFF) / 255f);
            }
        }

        if (bm.isRecycled()) {
            bm.recycle();
        }
        return imgData;
    }

//    public static Bitmap imageScale(Bitmap bitmap, int dst_w, int dst_h) {
//        int src_w = bitmap.getWidth();
//        int src_h = bitmap.getHeight();
//        float scale_w = ((float) dst_w) / src_w;
//        float scale_h = ((float) dst_h) / src_h;
//        Matrix matrix = new Matrix();
//        matrix.postScale(scale_w, scale_h);
//        Bitmap dstbmp = Bitmap.createBitmap(bitmap, 0, 0, src_w, src_h, matrix, true);
//        return dstbmp;
//    }


    //    compress picture
    public static Bitmap getScaleBitmap(String filePath) {
        BitmapFactory.Options opt = new BitmapFactory.Options();
        opt.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(filePath, opt);

        int bmpWidth = opt.outWidth;
        int bmpHeight = opt.outHeight;
        int maxSize = 500;
        // compress picture with inSampleSize
        opt.inSampleSize = 1;
        while (true) {
            if (bmpWidth / opt.inSampleSize < maxSize && bmpHeight / opt.inSampleSize < maxSize) {
                break;
            }
            opt.inSampleSize *= 2;
        }
        opt.inJustDecodeBounds = false;
        return BitmapFactory.decodeFile(filePath, opt);
    }
}
