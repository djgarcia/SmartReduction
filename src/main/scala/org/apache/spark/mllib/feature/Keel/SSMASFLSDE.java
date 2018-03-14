package org.apache.spark.mllib.feature.Keel;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class SSMASFLSDE implements Serializable {

    /*Own parameters of the algorithm*/

    private int tamPoblacion;
    private double nEval;
    private double pCross;
    private double pMut;
    private int kNeigh;
    private PrototypeSet trainingDataSet;

    private int PopulationSize;
    private int MaxIter;
    private int Strategy;


    private double tau[] = new double[4];
    private double Fl, Fu;

    private int iterSFGSS;
    private int iterSFHC;

    private boolean distanceEu;

    /**
     * Training input data.
     */
    private double datosTrain[][];

    /**
     * Training output data.
     */
    private int clasesTrain[];

    private int numFeat;
    private int nClases;

    private static MersenneTwister generador = new MersenneTwister();


    public SSMASFLSDE(double[][] data, int[] classes) {
        datosTrain = data;
        clasesTrain = classes;
        numFeat = datosTrain[0].length;
        leerConfiguracion();
    }

    private static double Rand() {
        return (generador.nextDouble());
    }

    private static int Randint(int low, int high) {
        return ((int) (low + (high - low) * generador.nextDouble()));
    }

    private static double Randdouble(double low, double high) {
        return (low + (high - low) * generador.nextDouble());
    }

    private void inic_vector_sin(int vector[], int without) {

        for (int i = 0; i < vector.length; i++)
            if (i != without)
                vector[i] = i;
    }

    private void desordenar_vector_sin(int vector[]) {
        int tmp, pos;
        for (int i = 0; i < vector.length - 1; i++) {
            pos = Randint(0, vector.length - 1);
            tmp = vector[i];
            vector[i] = vector[pos];
            vector[pos] = tmp;
        }
    }


    private PrototypeSet mutant(PrototypeSet population[], int actual, int mejor, double SFi) {


        PrototypeSet mutant = new PrototypeSet(population.length);
        PrototypeSet r1, r2, r3, r4, r5, resta, producto, resta2, producto2, result, producto3, resta3;

        //We need three differents solutions of actual

        int lista[] = new int[population.length];
        inic_vector_sin(lista, actual);
        desordenar_vector_sin(lista);

        // System.out.println("Lista = "+lista[0]+","+ lista[1]+","+lista[2]);

        r1 = population[lista[0]];
        r2 = population[lista[1]];
        r3 = population[lista[2]];
        r4 = population[lista[3]];
        r5 = population[lista[4]];

        switch (this.Strategy) {
            case 1: // ViG = Xr1,G + F(Xr2,G - Xr3,G) De rand 1
                resta = r2.restar(r3);
                producto = resta.mulEscalar(SFi);
                mutant = producto.sumar(r1);
                break;

            case 2: // Vig = Xbest,G + F(Xr2,G - Xr3,G)  De best 1
                resta = r2.restar(r3);
                producto = resta.mulEscalar(SFi);
                mutant = population[mejor].sumar(producto);
                break;

            case 3: // Vig = ... De rand to best 1
                resta = r1.restar(r2);
                resta2 = population[mejor].restar(population[actual]);

                producto = resta.mulEscalar(SFi);
                producto2 = resta2.mulEscalar(SFi);

                result = population[actual].sumar(producto);
                mutant = result.sumar(producto2);

                break;

            case 4: // DE best 2
                resta = r1.restar(r2);
                resta2 = r3.restar(r4);

                producto = resta.mulEscalar(SFi);
                producto2 = resta2.mulEscalar(SFi);

                result = population[mejor].sumar(producto);
                mutant = result.sumar(producto2);
                break;

            case 5: //DE rand 2
                resta = r2.restar(r3);
                resta2 = r4.restar(r5);

                producto = resta.mulEscalar(SFi);
                producto2 = resta2.mulEscalar(SFi);

                result = r1.sumar(producto);
                mutant = result.sumar(producto2);

                break;

            case 6: //DE rand to best 2
                resta = r1.restar(r2);
                resta2 = r3.restar(r4);
                resta3 = population[mejor].restar(population[actual]);

                producto = resta.mulEscalar(SFi);
                producto2 = resta2.mulEscalar(SFi);
                producto3 = resta3.mulEscalar(SFi);

                result = population[actual].sumar(producto);
                result = result.sumar(producto2);
                mutant = result.sumar(producto3);
                break;

        }

        mutant.applyThresholds();

        return mutant;
    }


    /**
     * Local Search Fitness Function
     */
    private double lsff(double Fi, double CRi, PrototypeSet population[], int actual, int mejor) {
        PrototypeSet resta, producto, mutant;
        PrototypeSet crossover;
        double FitnessFi = 0;


        //Mutation:
        mutant = new PrototypeSet(population[actual].size());
        mutant = mutant(population, actual, mejor, Fi);


        //Crossover
        crossover = new PrototypeSet(population[actual]);

        for (int j = 0; j < population[actual].size(); j++) { // For each part of the solution

            double randNumber = Randdouble(0, 1);

            if (randNumber < CRi) {
                crossover.set(j, mutant.get(j)); // Overwrite.
            }
        }


        // Compute fitness
        PrototypeSet nominalPopulation = new PrototypeSet();
        nominalPopulation.formatear(crossover);
        FitnessFi = classficationAccuracy1NN(nominalPopulation, trainingDataSet);

        return FitnessFi;
    }


    /**
     * SFGSS local Search.
     *
     * @param population
     * @return
     */
    private PrototypeSet SFGSS(PrototypeSet population[], int actual, int mejor, double CRi) {
        double a = 0.1, b = 1;

        double fi1 = 0, fi2 = 0, fitnessFi1 = 0, fitnessFi2 = 0;
        double phi = (1 + Math.sqrt(5)) / 5;
        double scaling;
        PrototypeSet crossover, resta, producto, mutant;

        for (int i = 0; i < this.iterSFGSS; i++) { // Computation budjet

            fi1 = b - (b - a) / phi;
            fi2 = a + (b - a) / phi;

            fitnessFi1 = lsff(fi1, CRi, population, actual, mejor);
            fitnessFi2 = lsff(fi2, CRi, population, actual, mejor);

            if (fitnessFi1 > fitnessFi2) {
                b = fi2;
            } else {
                a = fi1;
            }

        } // End While


        if (fitnessFi1 > fitnessFi2) {
            scaling = fi1;
        } else {
            scaling = fi2;
        }


        //Mutation:
        mutant = new PrototypeSet(population[actual].size());
        mutant = mutant(population, actual, mejor, scaling);

        //Crossover
        crossover = new PrototypeSet(population[actual]);

        for (int j = 0; j < population[actual].size(); j++) { // For each part of the solution

            double randNumber = Randdouble(0, 1);

            if (randNumber < CRi) {
                crossover.set(j, mutant.get(j)); // Overwrite.
            }
        }


        return crossover;
    }

    /**
     * SFHC local search
     *
     * @param actual
     * @param SFi
     * @return
     */

    private PrototypeSet SFHC(PrototypeSet population[], int actual, int mejor, double SFi, double CRi) {
        double fitnessFi1, fitnessFi2, fitnessFi3, bestFi;
        PrototypeSet crossover, resta, producto, mutant;
        double h = 0.5;


        for (int i = 0; i < this.iterSFHC; i++) { // Computation budjet

            fitnessFi1 = lsff(SFi - h, CRi, population, actual, mejor);
            fitnessFi2 = lsff(SFi, CRi, population, actual, mejor);
            fitnessFi3 = lsff(SFi + h, CRi, population, actual, mejor);

            if (fitnessFi1 >= fitnessFi2 && fitnessFi1 >= fitnessFi3) {
                bestFi = SFi - h;
            } else if (fitnessFi2 >= fitnessFi1 && fitnessFi2 >= fitnessFi3) {
                bestFi = SFi;
                h = h / 2; // H is halved.
            } else {
                bestFi = SFi;
            }

            SFi = bestFi;
        }


        //Mutation:
        mutant = new PrototypeSet(population[actual].size());
        mutant = mutant(population, actual, mejor, SFi);

        //Crossover
        crossover = new PrototypeSet(population[actual]);

        for (int j = 0; j < population[actual].size(); j++) { // For each part of the solution

            double randNumber = Randdouble(0, 1);

            if (randNumber < CRi) {
                crossover.set(j, mutant.get(j)); // Overwrite.
            }
        }


        return crossover;

    }

    private static double[] primitive(ArrayList<Double> array) {
        double[] target = new double[array.size() - 1];
        for (int i = 0; i < target.length; i++) {
            target[i] = array.get(i);
        }
        return target;
    }

    /**
     * Implements the 1NN algorithm
     *
     * @param current Prototype which the algorithm will find its nearest-neighbor.
     * @param dataSet Prototype set in which the algorithm will search.
     * @return Nearest prototype to current in the prototype set dataset.
     */
    private ArrayList<Double> _1nn(ArrayList<Double> current, PrototypeSet dataSet) {
        int indexNN = 0;

        double minDist = Double.POSITIVE_INFINITY;
        double currDist;
        int _size = dataSet.size();

        for (int i = 0; i < _size; i++) {
            ArrayList<Double> pi = dataSet.get(i);

            //currDist = Math.sqrt(squaredEuclideanDistance(pi, current));

            currDist = distancia(primitive(pi), primitive(current), true);

            if (currDist > 0) {
                if (currDist < minDist) {
                    minDist = currDist;
                    indexNN = i;
                }
            }
        }

        return dataSet.get(indexNN);
    }

    private double classficationAccuracy1NN(PrototypeSet training, PrototypeSet test) {
        int wellClassificated = 0;
        for (ArrayList<Double> p : test) {
            ArrayList<Double> nearestNeighbor = _1nn(p, training);
            if (p.get(p.size() - 1).equals(nearestNeighbor.get(nearestNeighbor.size() - 1))) {
                ++wellClassificated;
            }
        }
        return 100.0 * (wellClassificated / (double) test.size());
    }


    /**
     * Generate a reduced prototype set by the SADEGenerator method.
     *
     * @return Reduced set by SADEGenerator's method.
     */
    private PrototypeSet reduceSet(PrototypeSet initial) {
        System.out.print("\nThe algorithm  SSMA-SFLSDE is starting...\n Computing...\n");

        //Algorithm
        // First, we create the population, with PopulationSize.
        // like a prototypeSet's vector.

        PrototypeSet population[] = new PrototypeSet[PopulationSize];
        PrototypeSet mutation[] = new PrototypeSet[PopulationSize];
        PrototypeSet crossover[] = new PrototypeSet[PopulationSize];


        double ScalingFactor[] = new double[this.PopulationSize];
        double CrossOverRate[] = new double[this.PopulationSize]; // Inside of the Optimization process.
        double fitness[] = new double[PopulationSize];

        double fitness_bestPopulation[] = new double[PopulationSize];
        PrototypeSet bestParticle = new PrototypeSet();


        //Each particle must have   Particle Size %

        // First Stage, Initialization.

        PrototypeSet nominalPopulation;

        population[0] = new PrototypeSet(initial.clone());


        // Por si SSMA falla:

        if (population[0].size() < nClases) {
            int numberOfPrototypes = (int) Math.round(trainingDataSet.size() * 0.05);

            Collections.shuffle(trainingDataSet);
            population[0].addAll(trainingDataSet.subList(0, numberOfPrototypes));

            // red .95

            // Aseguro que al menos hay un representante de cada clase.
            PrototypeSet clases[] = new PrototypeSet[nClases];
            for (int i = 0; i < nClases; i++) {
                clases[i] = new PrototypeSet(trainingDataSet.getFromClass(i));

            }

            for (int i = 0; i < population[0].size(); i++) {
                for (int j = 0; j < nClases; j++) {
                    if (population[0].getFromClass(j).size() == 0 && clases[j].size() != 0) {

                        population[0].add(clases[j].getRandom());
                    }
                }
            }
        }


        nominalPopulation = new PrototypeSet();
        nominalPopulation.formatear(population[0]);

        fitness[0] = classficationAccuracy1NN(nominalPopulation, trainingDataSet);

        System.out.println("Best initial fitness = " + fitness[0]);

        for (int i = 1; i < PopulationSize; i++) {
            population[i] = new PrototypeSet();
            for (int j = 0; j < population[0].size(); j++) {
                ArrayList<Double> aux = new ArrayList<Double>(trainingDataSet.getFromClass(population[0].get(j).get(population[0].get(j).size() - 1)).getRandom());
                population[i].add(aux);
            }

            nominalPopulation = new PrototypeSet();
            nominalPopulation.formatear(population[i]);

            fitness[i] = classficationAccuracy1NN(population[i], trainingDataSet);   // PSOfitness
            fitness_bestPopulation[i] = fitness[i]; // Initially the same fitness.
        }


        //We select the best initial  particle
        double bestFitness = fitness[0];
        int bestFitnessIndex = 0;
        for (int i = 1; i < PopulationSize; i++) {
            if (fitness[i] > bestFitness) {
                bestFitness = fitness[i];
                bestFitnessIndex = i;
            }

        }

        // Initially the Scaling Factor and crossover for each Individual are randomly generated between 0 and 1.

        for (int i = 0; i < this.PopulationSize; i++) {
            ScalingFactor[i] = Randdouble(0, 1);
            CrossOverRate[i] = Randdouble(0, 1);
        }


        double randj[] = new double[5];


        for (int iter = 0; iter < MaxIter; iter++) { // Main loop

            for (int i = 0; i < PopulationSize; i++) {

                // Generate randj for j=1 to 5.
                for (int j = 0; j < 5; j++) {
                    randj[j] = Randdouble(0, 1);
                }


                if (i == bestFitnessIndex && randj[4] < tau[2]) {
                    // System.out.println("SFGSS applied");
                    //SFGSS
                    crossover[i] = SFGSS(population, i, bestFitnessIndex, CrossOverRate[i]);


                } else if (i == bestFitnessIndex && tau[2] <= randj[4] && randj[4] < tau[3]) {
                    //SFHC
                    //System.out.println("SFHC applied");
                    crossover[i] = SFHC(population, i, bestFitnessIndex, ScalingFactor[i], CrossOverRate[i]);

                } else {

                    // Fi update

                    if (randj[1] < tau[0]) {
                        ScalingFactor[i] = this.Fl + this.Fu * randj[0];
                    }

                    // CRi update

                    if (randj[3] < tau[1]) {
                        CrossOverRate[i] = randj[2];
                    }

                    // Mutation Operation.

                    mutation[i] = new PrototypeSet(population[i].size());

                    //Mutation:

                    mutation[i] = mutant(population, i, bestFitnessIndex, ScalingFactor[i]);

                    // Crossver Operation.

                    crossover[i] = new PrototypeSet(population[i]);

                    for (int j = 0; j < population[i].size(); j++) { // For each part of the solution

                        double randNumber = Randdouble(0, 1);

                        if (randNumber < CrossOverRate[i]) {
                            crossover[i].set(j, mutation[i].get(j)); // Overwrite.
                        }
                    }


                }


                // Fourth: Selection Operation.

                nominalPopulation = new PrototypeSet();
                nominalPopulation.formatear(population[i]);
                fitness[i] = classficationAccuracy1NN(nominalPopulation, trainingDataSet);

                nominalPopulation = new PrototypeSet();
                nominalPopulation.formatear(crossover[i]);

                double trialVector = classficationAccuracy1NN(nominalPopulation, trainingDataSet);


                if (trialVector > fitness[i]) {
                    population[i] = new PrototypeSet(crossover[i]);
                    fitness[i] = trialVector;
                }

                if (fitness[i] > bestFitness) {
                    bestFitness = fitness[i];
                    bestFitnessIndex = i;
                }


            }


        }


        nominalPopulation = new PrototypeSet();
        nominalPopulation.formatear(population[bestFitnessIndex]);
        System.err.println("\n% de acierto en training Nominal " + classficationAccuracy1NN(nominalPopulation, trainingDataSet));

        //  nominalPopulation.print();


        return nominalPopulation;
    }


    /* MEzcla de algoritmos */
    public LabeledPoint[] ejecutar() {

        int i, j, l;
        double conjData[][];
        int clasesS[];
        int nSel = 0;
        Cromosoma poblacion[];
        double ev = 0;
        double dMatrix[][];
        int sel1, sel2, comp1, comp2;
        Cromosoma hijos[];
        double umbralOpt;
        boolean veryLarge;
        double GAeffort = 0, LSeffort = 0, temporal;
        double fAcierto = 0, fReduccion = 0;
        int contAcierto = 0, contReduccion = 0;

        long tiempo = System.currentTimeMillis();

        /*Getting the number of different classes*/
        nClases = 0;
        for (i = 0; i < clasesTrain.length; i++)
            if (clasesTrain[i] > nClases)
                nClases = clasesTrain[i];
        nClases++;

        if (datosTrain.length > 9000) {
            veryLarge = true;
        } else {
            veryLarge = false;
        }

        if (veryLarge == false) {
            /*Construct a distance matrix of the instances*/
            dMatrix = new double[datosTrain.length][datosTrain.length];
            for (i = 0; i < dMatrix.length; i++) {
                for (j = i + 1; j < dMatrix[i].length; j++) {
                    dMatrix[i][j] = distancia(datosTrain[i], datosTrain[j], distanceEu);
                }
            }
            for (i = 0; i < dMatrix.length; i++) {
                dMatrix[i][i] = Double.POSITIVE_INFINITY;
            }
            for (i = 0; i < dMatrix.length; i++) {
                for (j = i - 1; j >= 0; j--) {
                    dMatrix[i][j] = dMatrix[j][i];
                }
            }
        } else {
            dMatrix = null;
        }

        /*Random inicialization of the population*/
        poblacion = new Cromosoma[tamPoblacion];
        for (i = 0; i < tamPoblacion; i++)
            poblacion[i] = new Cromosoma(kNeigh, datosTrain.length, dMatrix, datosTrain, distanceEu);

        /*Initial evaluation of the population*/
        for (i = 0; i < tamPoblacion; i++) {
            poblacion[i].evaluacionCompleta(nClases, kNeigh, clasesTrain);
        }

        umbralOpt = 0;

        /*Until stop condition*/
        while (ev < nEval) {

            Arrays.sort(poblacion);

            if (fAcierto >= (double) poblacion[0].getFitnessAc() * 100.0 / (double) datosTrain.length) {
                contAcierto++;
            } else {
                contAcierto = 0;
            }
            fAcierto = (double) poblacion[0].getFitnessAc() * 100.0 / (double) datosTrain.length;


            if (fReduccion >= (1.0 - ((double) poblacion[0].genesActivos() / (double) datosTrain.length)) * 100.0) {
                contReduccion++;
            } else {
                contReduccion = 0;
            }
            fReduccion = (1.0 - ((double) poblacion[0].genesActivos() / (double) datosTrain.length)) * 100.0;

            if (contReduccion >= 10 || contAcierto >= 10) {
                double random = generador.nextDouble();
                if (random >= 0.5) {
                    if (contAcierto >= 10) {
                        contAcierto = 0;
                        umbralOpt++;
                    } else {
                        contReduccion = 0;
                        umbralOpt--;
                    }
                } else {
                    if (contReduccion >= 10) {
                        contReduccion = 0;
                        umbralOpt--;
                    } else {
                        contAcierto = 0;
                        umbralOpt++;
                    }
                }
            }

            /*Binary tournament selection*/
            comp1 = Randint(0, tamPoblacion - 1);
            do {
                comp2 = Randint(0, tamPoblacion - 1);
            } while (comp2 == comp1);

            if (poblacion[comp1].getFitness() > poblacion[comp2].getFitness()) {
                sel1 = comp1;
            } else {
                sel1 = comp2;
            }
            comp1 = Randint(0, tamPoblacion - 1);
            do {
                comp2 = Randint(0, tamPoblacion - 1);
            } while (comp2 == comp1);
            if (poblacion[comp1].getFitness() > poblacion[comp2].getFitness()) {
                sel2 = comp1;
            } else {
                sel2 = comp2;
            }

            hijos = new Cromosoma[2];
            hijos[0] = new Cromosoma(kNeigh, poblacion[sel1], poblacion[sel2], pCross, datosTrain.length);
            hijos[1] = new Cromosoma(kNeigh, poblacion[sel2], poblacion[sel1], pCross, datosTrain.length);
            hijos[0].mutation(kNeigh, pMut, dMatrix, datosTrain, distanceEu);
            hijos[1].mutation(kNeigh, pMut, dMatrix, datosTrain, distanceEu);

            /*Evaluation of offsprings*/
            hijos[0].evaluacionCompleta(nClases, kNeigh, clasesTrain);
            hijos[1].evaluacionCompleta(nClases, kNeigh, clasesTrain);

            ev += 2;
            GAeffort += 2;
            temporal = ev;
            if (hijos[0].getFitness() > poblacion[tamPoblacion - 1].getFitness() || Rand() < 0.0625) {
                ev += hijos[0].optimizacionLocal(nClases, kNeigh, clasesTrain, dMatrix, umbralOpt, datosTrain, distanceEu);
            }
            if (hijos[1].getFitness() > poblacion[tamPoblacion - 1].getFitness() || Rand() < 0.0625) {
                ev += hijos[1].optimizacionLocal(nClases, kNeigh, clasesTrain, dMatrix, umbralOpt, datosTrain, distanceEu);
            }

            LSeffort += (ev - temporal);

            /*Replace the two worst*/
            if (hijos[0].getFitness() > poblacion[tamPoblacion - 1].getFitness()) {
                poblacion[tamPoblacion - 1] = new Cromosoma(kNeigh, datosTrain.length, hijos[0]);
            }
            if (hijos[1].getFitness() > poblacion[tamPoblacion - 2].getFitness()) {
                poblacion[tamPoblacion - 2] = new Cromosoma(kNeigh, datosTrain.length, hijos[1]);
            }

        }

        Arrays.sort(poblacion);

        nSel = poblacion[0].genesActivos();

        /*Construction of S set from the best cromosome*/
        conjData = new double[nSel][datosTrain[0].length];
        clasesS = new int[nSel];
        for (i = 0, l = 0; i < datosTrain.length; i++) {
            if (poblacion[0].getGen(i)) { //the instance must be copied to the solution
                for (j = 0; j < datosTrain[i].length; j++) {
                    conjData[l][j] = datosTrain[i][j];
                }
                clasesS[l] = clasesTrain[i];
                l++;
            }
        }

        System.out.println("SSMA " + (double) (System.currentTimeMillis() - tiempo) / 1000.0 + "s");

        PrototypeSet training = new PrototypeSet();

        for (int k = 0; k < conjData.length; k++) {
            ArrayList<Double> lista = new ArrayList<Double>();
            for (int m = 0; m < conjData[k].length; m++) {
                lista.add(conjData[k][m]);
            }
            lista.add((double) clasesS[k]);

            training.add(lista);
        }

        trainingDataSet = new PrototypeSet();

        for (int k = 0; k < datosTrain.length; k++) {
            ArrayList<Double> lista = new ArrayList<Double>();
            for (int m = 0; m < datosTrain[k].length; m++) {
                lista.add(datosTrain[k][m]);
            }
            lista.add((double) clasesTrain[k]);

            trainingDataSet.add(lista);
        }

        PrototypeSet SADE = reduceSet(training); // LLAMO al SADE

        LabeledPoint finalData[] = new LabeledPoint[SADE.size()];

        for (j = 0; j < SADE.size(); j++) {
            ArrayList<Double> instancia = SADE.get(j);
            double clase = instancia.get(instancia.size() - 1);
            instancia.remove(instancia.size() - 1);
            Double[] inst = instancia.toArray(new Double[instancia.size()]);
            double[] keepThis = ArrayUtils.toPrimitive(inst);

            finalData[j] = new LabeledPoint(clase, Vectors.dense(keepThis));
        }

        //Copy the test input file to the output test file
        System.out.println("Time elapse:" + (double) (System.currentTimeMillis() - tiempo) / 1000.0 + "s");

        return finalData;
    }

    private void leerConfiguracion() {
        tamPoblacion = 30;
        nEval = 10000;
        pCross = 0.5;
        pMut = 0.001;
        kNeigh = 1;
        distanceEu = true;
        PopulationSize = 40;
        this.MaxIter = 500;
        this.iterSFGSS = 8;
        this.iterSFHC = 20;
        this.Fl = 0.1;
        this.Fu = 0.9;
        tau = new double[4];
        this.tau[0] = 0.1;
        this.tau[1] = 0.1;
        this.tau[2] = 0.03;
        this.tau[3] = 0.07;
        this.Strategy = 3;
    }

    private static double distancia(double ej1[], double ej2[], boolean Euc) {
        int i;
        double suma = 0;

        if (Euc == true) {
            for (i = 0; i < ej1.length; i++) {
                suma += (ej1[i] - ej2[i]) * (ej1[i] - ej2[i]);
            }
            suma = Math.sqrt(suma);
        }
        return suma;
    }
}
