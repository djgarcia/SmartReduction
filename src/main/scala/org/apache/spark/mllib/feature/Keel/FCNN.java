package org.apache.spark.mllib.feature.Keel;

import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Vector;

public class FCNN implements Serializable {

    /**
     * Training input data.
     */
    private double datosTrain[][];

    /**
     * Training output data.
     */
    private int clasesTrain[];

    /*Own parameters of the algorithm*/
    private int k;

    /**
     * Default builder. Construct the algoritm by using the superclass builder.
     * <p>
     * //@param ficheroScript Configuration script
     */
    public FCNN(double[][] data, int[] classes, int knn) {
        datosTrain = data;
        clasesTrain = classes;
        k = knn;
    }

    /**
     * Executes the algorithm
     */
    public LabeledPoint[] ejecutar() {

        LabeledPoint conjData[];
        int S[];
        int i, j, l, m;
        int nClases;
        int pos;
        int tamS;
        int nearest[][];
        Vector<Integer> deltaS = new Vector<Integer>();
        double centroid[];
        int nCentroid;
        double dist, minDist;
        int rep[];
        boolean insert;
        int votes[];
        int max;

        long tiempo = System.currentTimeMillis();

        /*Getting the number of different classes*/
        nClases = 0;
        for (i = 0; i < clasesTrain.length; i++)
            if (clasesTrain[i] > nClases)
                nClases = clasesTrain[i];
        nClases++;

        if (nClases < 2) {
            System.err.println("Input dataset has only one class");
            nClases = 0;
        }

        nearest = new int[datosTrain.length][k];
        for (i = 0; i < datosTrain.length; i++) {
            Arrays.fill(nearest[i], -1);
        }

        /*Inicialization of the candidates set*/
        S = new int[datosTrain.length];
        for (i = 0; i < S.length; i++)
            S[i] = Integer.MAX_VALUE;
        tamS = 0;

        /*Inserting an element of each class*/
        centroid = new double[datosTrain[0].length];
        for (i = 0; i < nClases; i++) {
            nCentroid = 0;
            Arrays.fill(centroid, 0);
            for (j = 0; j < datosTrain.length; j++) {
                if (clasesTrain[j] == i) {
                    for (l = 0; l < datosTrain[j].length; l++) {
                        centroid[l] += datosTrain[j][l];
                    }
                    nCentroid++;
                }
            }
            for (j = 0; j < centroid.length; j++) {
                centroid[j] /= (double) nCentroid;
            }
            pos = -1;
            minDist = Double.POSITIVE_INFINITY;
            for (j = 0; j < datosTrain.length; j++) {
                if (clasesTrain[j] == i) {
                    dist = distancia(centroid, datosTrain[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        pos = j;
                    }
                }
            }
            if (pos >= 0)
                deltaS.add(pos);
        }

        /*Algorithm body*/
        rep = new int[datosTrain.length];
        votes = new int[nClases];
        while (deltaS.size() > 0) {

            for (i = 0; i < deltaS.size(); i++) {
                S[tamS] = deltaS.elementAt(i);
                tamS++;
            }
            Arrays.sort(S);

            Arrays.fill(rep, -1);

            for (i = 0; i < datosTrain.length; i++) {
                if (Arrays.binarySearch(S, i) < 0) {
                    for (j = 0; j < deltaS.size(); j++) {
                        insert = false;
                        for (l = 0; l < nearest[i].length && !insert; l++) {
                            if (nearest[i][l] < 0) {
                                nearest[i][l] = deltaS.elementAt(j);
                                insert = true;
                            } else {
                                if (distancia(datosTrain[nearest[i][l]], datosTrain[i]) > distancia(datosTrain[i], datosTrain[deltaS.elementAt(j)])) {
                                    for (m = k - 1; m >= l + 1; m--) {
                                        nearest[i][m] = nearest[i][m - 1];
                                    }
                                    nearest[i][l] = deltaS.elementAt(j);
                                    insert = true;
                                }
                            }
                        }
                    }

                    Arrays.fill(votes, 0);
                    for (j = 0; j < nearest[i].length; j++) {
                        if (nearest[i][j] >= 0) {
                            votes[clasesTrain[nearest[i][j]]]++;
                        }
                    }
                    max = votes[0];
                    pos = 0;
                    for (j = 1; j < votes.length; j++) {
                        if (votes[j] > max) {
                            max = votes[j];
                            pos = j;
                        }
                    }
                    if (clasesTrain[i] != pos) {
                        for (j = 0; j < nearest[i].length; j++) {
                            if (nearest[i][j] >= 0) {
                                if (rep[nearest[i][j]] < 0) {
                                    rep[nearest[i][j]] = i;
                                } else {
                                    if (distancia(datosTrain[nearest[i][j]], datosTrain[i]) < distancia(datosTrain[nearest[i][j]], datosTrain[rep[nearest[i][j]]])) {
                                        rep[nearest[i][j]] = i;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            deltaS.removeAllElements();

            for (i = 0; i < tamS; i++) {
                if (rep[S[i]] >= 0 && !deltaS.contains(rep[S[i]]))
                    deltaS.add(rep[S[i]]);
            }
        }

        /*Construction of the S set from the previous vector S*/
        conjData = new LabeledPoint[tamS];
        for (j = 0; j < tamS; j++) {
            conjData[j] = new LabeledPoint(clasesTrain[S[j]], Vectors.dense(datosTrain[S[j]]));
        }

        System.out.println("FCNN " + (double) (System.currentTimeMillis() - tiempo) / 1000.0 + "s");

        return conjData;
    }

    /**
     * Calculates the Euclidean distance between two instances
     *
     * @param ej1 First instance
     * @param ej2 Second instance
     * @return The Euclidean distance
     */
    private static double distancia(double ej1[], double ej2[]) {

        int i;
        double suma = 0;

        for (i = 0; i < ej1.length; i++) {
            suma += (ej1[i] - ej2[i]) * (ej1[i] - ej2[i]);
        }
        suma = Math.sqrt(suma);

        return suma;
    }
}
