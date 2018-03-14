/***********************************************************************

 This file is part of KEEL-software, the Data Mining tool for regression,
 classification, clustering, pattern mining and so on.

 Copyright (C) 2004-2010

 F. Herrera (herrera@decsai.ugr.es)
 L. Sánchez (luciano@uniovi.es)
 J. Alcalá-Fdez (jalcala@decsai.ugr.es)
 S. García (sglopez@ujaen.es)
 A. Fernández (alberto.fernandez@ujaen.es)
 J. Luengo (julianlm@decsai.ugr.es)

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see http://www.gnu.org/licenses/

 **********************************************************************/

package org.apache.spark.mllib.feature.Keel;

import java.io.Serializable;
import java.util.*;

/**
 * Represents a prototype set.
 *
 * @author diegoj and Isaac
 */
public class PrototypeSet extends ArrayList<ArrayList<Double>> implements Serializable {
    /**
     * Empty constructor
     */
    public PrototypeSet() {
        super();
    }

    /**
     * Constructs a void set with a number of elements.
     *
     * @param numberOfElements Maximum inicial capacity.
     */
    public PrototypeSet(int numberOfElements) {
        super(numberOfElements);
    }

    /**
     * Copy constructor. NOTE: soft-copy.
     *
     * @param original Original set to be copied.
     */
    public PrototypeSet(PrototypeSet original) {
        super(original.size());
        /*int _size = original.size();
        for(int i=0; i<_size; ++i)
            add(new Prototype(original.get(i)));*/
        for (ArrayList<Double> p : original)
            add(p);
    }

    /**
     * Get a random prototype
     *
     * @return Random prototype of the set
     */
    public ArrayList<Double> getRandom() {
        Random rand = new Random();
        //return get(((int) (0 + (((size()-1)+1) - 0) * rand.nextFloat())));
        return get(rand.nextInt(size()-1));
    }

    /**
     * Select all the prototypes of a specific class. The class must be a valid!.
     *
     * @param _class Choosen class .
     * @return A prototype set which contains all the prototypes of the original set that are of the choosen class.
     */
    public PrototypeSet getFromClass(double _class) {
        PrototypeSet selected = new PrototypeSet(size() / 2);
        for (ArrayList<Double> p : this)
            if (p.get(p.size() - 1) == _class)
                selected.add(p);
        return selected;
    }

    /**
     * Override of the clone function
     *
     * @return A new object hard-new-copy of the caller.
     */
    public PrototypeSet clone() {
        //return new PrototypeSet(this);
        PrototypeSet copy = new PrototypeSet(this.size());
        for (ArrayList<Double> p : this)
            copy.add(new ArrayList<Double>(p));
        return copy;
    }

    /**
     * Hard-copy of the prototype set.
     *
     * @return A new object hard-new-copy of the caller.
     */
    public PrototypeSet copy() {
        return this.clone();
    }

    /**
     * Converts data set into a Keelish-String.
     *
     * @return String with the Keel data file representation of the prototype set
     */
    public String asKeelDataFileString() {
        String text = "";

        int n_attributes = get(0).size() - 1;
        for (ArrayList<Double> q : this) {
            //ArrayList<Double> q = p.denormalize(); //TOKADO PARA NO NORMALIZAR
//            for (int i = 0; i < n_attributes; ++i) {
//                double q_i = q.get(i); //
//                if (Prototype.getTypeOfAttribute(i) == Prototype.INTEGER)
//                    text += Math.round(q_i) + ", "; // ERROR de DIEGO!?
//                else if (Prototype.getTypeOfAttribute(i) == Prototype.DOUBLE)
//                    text += q_i + ", ";
//
//            }

            text += q.get(q.size() - 1) + "\n";
        }
        return text;
    }

    /**
     * Add the elements of a set.
     *
     * @param other Set with the elements to include.
     */
    public void add(PrototypeSet other) {
        for (ArrayList<Double> p : other)
            this.add(p);
    }

    /**
     * Compute the squared euclidean distance between two prototypes.
     *
     * @param one One prototype.
     * @param two Other prototype.
     * @return squared euclidean distance between one and two.
     */
    private static double squaredEuclideanDistance(ArrayList<Double> one, ArrayList<Double> two) {
        one.remove(one.size() - 1);
        two.remove(two.size() - 1);
        final Double[] oneInputs = one.toArray(new Double[one.size()]);
        final Double[] twoInputs = two.toArray(new Double[two.size()]);
        //final int _size = one.numberOfInputs();
        double acc = 0.0;
        for (int i = 0; i < one.size(); i++) {
            acc += (oneInputs[i] - twoInputs[i]) * (oneInputs[i] - twoInputs[i]);
        }
        return acc;
    }

    /**
     * Returns the smallest distance between uno and all prototypes of the particle.
     *
     * @param uno prototype given to compute the distance with.
     * @return the smallest distance between uno and all prototypes of the particle.
     */

    public double minDist(ArrayList<Double> uno) {

        double min = 999999999;
        //Prototype pMin = null;
        for (ArrayList<Double> p : this) {
            double current = Math.sqrt(squaredEuclideanDistance(p, uno));
            if (current < min) {
                min = current;

            }
        }

        return min;
    }

    /**
     * Change values of the prototypes that are not in the values domain.
     */
    public void applyThresholds() {
        for (ArrayList<Double> p : this)
            for (int i = 0; i < p.size(); ++i) {
                if (p.get(i) > 1)
                    p.set(i, 1.0);
                else if (p.get(i) < 0)
                    p.set(i, 0.0);
            }
    }

    /**********************************************
     *
     * FUNCIONES PARA DIFFERENTIAL EVOLUTION.
     * ********************************************
     */

    /**
     * Performs add operation between two prototypes.
     *
     * @param other A protype to be added to the implicit parameter.
     * @return A prototype which inputs are the sum of another two, and outputs are a copy of implicit-ones.
     */
    private ArrayList<Double> add(ArrayList<Double> one, ArrayList<Double> other) {
        int numInputs = one.size();
        Double[] _inputs = new Double[numInputs];

        for (int i = 0; i < numInputs-1; ++i)
            _inputs[i] = one.get(i) + other.get(i);

        _inputs[numInputs-1] = one.get(numInputs-1);

        ArrayList<Double> returned = new ArrayList<Double>();
        returned.addAll(Arrays.asList(_inputs));

        return returned;
    }


    /**
     * Sums two prototype sets, element by element. They should have the same dimension.
     *
     * @param other the prototype to sum.
     * @return the prototype set resulting from the operation. Null if the sets have not the same size.
     */
    public PrototypeSet sumar(PrototypeSet other) {
        PrototypeSet suma = new PrototypeSet();

        if (this.size() == other.size()) {
            for (int i = 0; i < this.size(); i++) {
                suma.add(add(this.get(i), other.get(i)));
            }
        } else {
            return null;
        }

        return suma;
    }

    /**
     * Performs substract operation between two prototypes.
     *
     * @param other A protype to be substract to the implicit parameter.
     * @return A prototype which inputs are the difference of another two, and outputs are a copy of implicit-ones.
     */
    private ArrayList<Double> sub(ArrayList<Double> one, ArrayList<Double> other) {
        int numInputs = one.size();
        Double[] _inputs = new Double[numInputs];

        for (int i = 0; i < numInputs-1; ++i)
            _inputs[i] = one.get(i) - other.get(i);

        _inputs[numInputs-1] = one.get(numInputs-1);

        ArrayList<Double> returned = new ArrayList<Double>();
        returned.addAll(Arrays.asList(_inputs));

        return returned;
    }

    /**
     * Subtracts two prototype sets, element by element. They should have the same dimension.
     *
     * @param other the prototype to subtract.
     * @return the prototype set resulting from the operation. Null if the sets have not the same size.
     */
    public PrototypeSet restar(PrototypeSet other) {
        PrototypeSet resta = new PrototypeSet();

        if (this.size() == other.size()) {
            for (int i = 0; i < this.size(); i++) {
                resta.add(sub(this.get(i), other.get(i)));
            }
        } else {
            return null;
        }

        return resta;
    }

    /**
     * Performs product operation between one prototype and a double.
     *
     * @param weight Constant to be multiplied to each sum.
     * @return A prototype which inputs product with a weight.
     */
    private ArrayList<Double> mul(ArrayList<Double> one, double weight) {
        int numInputs = one.size();
        Double[] _inputs = new Double[numInputs];

        for (int i = 0; i < numInputs-1; ++i)
            _inputs[i] = weight * (one.get(i));

        _inputs[numInputs-1] = one.get(numInputs-1);

        ArrayList<Double> returned = new ArrayList<Double>();
        returned.addAll(Arrays.asList(_inputs));

        return returned;
    }

    /**
     * Multiply the set by a number given.
     *
     * @param escalar number to multiply
     * @return the prototype set resulting from the operation.
     */
    public PrototypeSet mulEscalar(double escalar) {
        PrototypeSet result = new PrototypeSet();

        for (int i = 0; i < this.size(); i++) {
            result.add(mul(this.get(i), escalar));
        }

        return result;
    }

    /**
     * Adds the elements of a reset set given.
     *
     * @param initial given set.
     */
    public void formatear(PrototypeSet initial) {
        new PrototypeSet();

        for (int i = 0; i < initial.size(); i++) {
            //System.out.println(initial.get(i));
            ArrayList<Double> formateado = new ArrayList<Double>(initial.get(i));
            this.add(formateado);
        }
    }
}
