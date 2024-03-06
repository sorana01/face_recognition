/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Trace;
import android.util.Log;
import android.util.Pair;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.storage.FileDownloadTask;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.DetectorActivity;
import org.tensorflow.lite.examples.detection.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel
        implements SimilarityClassifier {

  private static final String FileName = "images";
  private final String TAG = "Class TFLiteObjectDetectionAPIModel :";

  private static final Logger LOGGER = new Logger();

  //private static final int OUTPUT_SIZE = 512;
  private static final int OUTPUT_SIZE = 192;

  // Only return this many results.
  private static final int NUM_DETECTIONS = 1;

  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;

  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private float[][] embeddings;

  private ByteBuffer imgData;

  private Interpreter tfLite;

// Face Mask Detector Output
  private float[][] output;

  private HashMap<String, Recognition> registered = new HashMap<>();

  public void registerOld(String name, Recognition rec) {
      registered.put(name, rec);
      LOGGER.i("Registered new face: " + name + " ," + rec);
  }

  public void register(String name, Recognition rec, DetectorActivity det) {
    registered.put(name, rec);
    LOGGER.i("Registered new face: " + name + " ," + rec);

    byte[] bytes=null;
    try {

      //  file.createNewFile();
      //write the bytes in file
      {
        Gson gson = new Gson();


        File localFile = new File(det.getFilesDir(),FileName);
        FileOutputStream fileOutputStream = new FileOutputStream(localFile);

        Type type = new TypeToken<HashMap<String, Recognition>>(){}.getType();
        String toStoreObject = gson.toJson(registered,type);

        ObjectOutputStream o = new ObjectOutputStream(fileOutputStream);
        o.writeObject(toStoreObject);
        //o.writeObject(registered);

        o.close();
        /* 26 */
        fileOutputStream.close();

        Toast.makeText(det.getApplicationContext(), "save file completed.", Toast.LENGTH_LONG ).show();

        Log.d("Clique AQUI","Clique AQUI file created: " );
        ///     file.delete();
        Log.d("Clique AQUI","Clique AQUI delete " );
      }

      FirebaseStorage storage = FirebaseStorage.getInstance();
      StorageReference storageRef = storage.getReference();
      StorageReference test2 = storageRef.child(FileName);
      //test2.delete();
      //test2.putStream();

      Uri file = Uri.fromFile(new File(det.getFilesDir(),FileName));


      test2.putFile(file)
              .addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                  // Get a URL to the uploaded content
                  //Uri downloadUrl = taskSnapshot.get();
                  Toast.makeText(det.getApplicationContext(), "Upload Completed.", Toast.LENGTH_LONG ).show();

                }
              })
              .addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception exception) {
                  // Handle unsuccessful uploads
                  // ...
                  Toast.makeText(det.getApplicationContext(), "Upload Failure.", Toast.LENGTH_LONG ).show();
                }
              });

      Log.d("Clique AQUI","Clique Aqui Enviou ");



    }catch (Exception e){


      Log.d("Clique AQUI","Clique AQUI file created: " + e.toString());

      //Log.d("Clique AQUI","Clique AQUI file created: " + bytes.length);
      Toast.makeText(det.getApplicationContext(), e.getMessage(), Toast.LENGTH_LONG ).show();

    }
  }

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static SimilarityClassifier createOld(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {

    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    InputStream labelsInput = assetManager.open(actualFilename);
    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }

  public static SimilarityClassifier create(
          final AssetManager assetManager,
          final String modelFilename,
          final String labelFilename,
          final int inputSize,
          final boolean isQuantized,
          DetectorActivity det)
          throws IOException {

    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();


    try {
      //Toast.makeText(det.getApplicationContext(), "name is null", Toast.LENGTH_LONG ).show();

      FirebaseStorage storage = FirebaseStorage.getInstance();
      StorageReference storageRef = storage.getReference();
      StorageReference test2 = storageRef.child(FileName);

      File localFile = File.createTempFile("Student", "txt");
      //File localFile = new File(det.getFilesDir(),"test2.txt");
      test2.getFile(localFile).addOnSuccessListener(new OnSuccessListener<FileDownloadTask.TaskSnapshot>() {
        @Override
        public void onSuccess(FileDownloadTask.TaskSnapshot taskSnapshot) {

          try {

            Gson gson = new Gson();
            ObjectInputStream i = new ObjectInputStream(new FileInputStream(localFile));
            //HashMap<String, Recognition> registeredl = (HashMap<String, Recognition>) i.readObject();

            Type type = new TypeToken<HashMap<String, Recognition>>(){}.getType();
            HashMap<String, Recognition> registeredl = gson.fromJson((String)i.readObject(), type);
            //HashMap<String, Recognition> registeredl = (HashMap<String, Recognition>) i.readObject();

            if (registeredl != null){
              d.registered = registeredl;
            }
            i.close();

            Toast.makeText(det.getApplicationContext(), "โหลดข้อมูลเรียบร้อย.", Toast.LENGTH_LONG ).show();
            Log.d("Clique AQUI", "Clique Aqui Adicionado " + registeredl.size());

          } catch (Exception e) {
            Log.d("Clique AQUI", "Clique Aqui erro " + e.toString());
            Toast.makeText(det.getApplicationContext(), "Exception 1" + e.getMessage(), Toast.LENGTH_LONG ).show();
          }
        }
      }).addOnFailureListener(new OnFailureListener() {
        @Override
        public void onFailure(@NonNull Exception exception) {
          Log.d("Clique AQUI", "Clique Aqui erro " + exception.toString());
          Toast.makeText(det.getApplicationContext(), "Exception 2 " + exception.getMessage(), Toast.LENGTH_LONG ).show();
        }
      });


    } catch (Exception e) {

      Log.d("Clique AQUI", "Clique AQUI file created: " + e.toString());
    }



    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    InputStream labelsInput = assetManager.open(actualFilename);
    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
// Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }

  // looks for the nearest embedding in the dataset (using L2 norm)
  // and returns the pair <id, distance>
  private Pair<String, Float> findNearestOld(float[] emb) {

    Pair<String, Float> ret = null;
    for (Map.Entry<String, Recognition> entry : registered.entrySet()) {
      final String name = entry.getKey();
      final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

      float distance = 0;
      for (int i = 0; i < emb.length; i++) {
        float diff = emb[i] - knownEmb[i];
        distance += diff*diff;
      }
      distance = (float) Math.sqrt(distance);
      if (ret == null || distance < ret.second) {
        ret = new Pair<>(name, distance);
      }
    }

    return ret;

  }

  private Pair<String, Float> findNearest(float[] emb) {

    Gson gson = new Gson();

    Pair<String, Float> ret = null;

    for (Map.Entry<String, Recognition> entry : registered.entrySet()) {
      String name = entry.getKey();

      float distance = 0;
      try {

        // original code
        //final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

        // -------------------- MODIFY --------------------------------------------------------------/
        float[][] knownEmb2d = gson.fromJson(entry.getValue().getExtra().toString(), float[][].class);
        final float[] knownEmb = knownEmb2d[0];

        for (int i = 0; i < emb.length; i++) {
          float diff = emb[i] - knownEmb[i];
          distance += diff * diff;
        }
      } catch (Exception e) {
        //Toast.makeText(context, e.getMessage(), Toast.LENGTH_LONG ).show();
        Log.e("findNearest", e.getMessage());
      }
      distance = (float) Math.sqrt(distance);
      if (ret == null || distance < ret.second) {
        ret = new Pair<>(name, distance);
      }
    }

    return ret;
  }


  // retrieve the embeddings and if necessary, store them into the recognition result
  // when we have the embedding, we look for the nearest neighbour embedding into dataset with linear search
  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap, boolean storeExtra) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    // Prepare the image data suitable for input to TensorFlow model
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");

    Object[] inputArray = {imgData};
    Trace.endSection();

// Here outputMap is changed to fit the Face Mask detector
    Map<Integer, Object> outputMap = new HashMap<>();

    embeddings = new float[1][OUTPUT_SIZE];
    outputMap.put(0, embeddings);


    // Run the inference call.
    Trace.beginSection("run");

    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();


    float distance = Float.MAX_VALUE;
    String id = "0";
    String label = "Unknown";

    // if not empty search
    if (registered.size() > 0) {
        LOGGER.i("dataset SIZE: " + registered.size());
        // looks for the nearest neighbour
        final Pair<String, Float> nearest = findNearest(embeddings[0]);
        if (nearest != null) {

            final String name = nearest.first;
            label = name;
            distance = nearest.second;

            LOGGER.i("nearest: " + name + " - distance: " + distance);


        }
    }
    //    else Label as unknown if not found??



    final int numDetectionsOutput = 1;
    final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
    Recognition rec = new Recognition(
            id,
            label,
            distance,
            new RectF());

    // either Unknown or name of recognized from findNearest()
    recognitions.add( rec );

    // storeExtra bool true = new face to add
    if (storeExtra) {
        rec.setExtra(embeddings);
    }

    Trace.endSection();   //recognizeImage
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }
}
