package de.hnu;

import com.ugos.jiprolog.engine.JIPDebugger;
import com.ugos.jiprolog.engine.JIPEngine;
import com.ugos.jiprolog.engine.JIPErrorEvent;
import com.ugos.jiprolog.engine.JIPEvent;
import com.ugos.jiprolog.engine.JIPEventListener;
import com.ugos.jiprolog.engine.JIPTerm;

public class PrologRunner implements JIPEventListener {

    @Override
    public void errorNotified(JIPErrorEvent e) {
        synchronized(e.getSource()) {
            int nQueryHandle = e.getQueryHandle();
            JIPEngine jip = e.getSource();            
            jip.closeQuery(nQueryHandle);
        }
    }

    @Override
    public void solutionNotified(JIPEvent e) {
        synchronized(e.getSource()) {
            int nQueryHandle = e.getQueryHandle();            
            System.out.println(e.getTerm());

            JIPEngine jip = e.getSource();            
            jip.nextSolution(nQueryHandle);
        }
    }

    @Override
    public void closeNotified(JIPEvent e) {
        synchronized(e.getSource()) {
            e.getSource().notify();
        }
        System.exit(0);
    }

    @Override
    public void moreNotified(JIPEvent arg0) {

    }

    @Override
    public void termNotified(JIPEvent arg0) {
    }

    @Override
    public void openNotified(JIPEvent arg0) {
    }

    @Override
    public void endNotified(JIPEvent e) {
        synchronized(e.getSource()) {
            int nQueryHandle = e.getQueryHandle();
            JIPEngine jip = e.getSource();            
            jip.closeQuery(nQueryHandle);
        }
    }

    public static void main(String args[]) {
        try {
            JIPDebugger.debug = true;

            // New instance of prolog engine
            JIPEngine jip = new JIPEngine();                
            System.out.println(jip.getVersion());
        
            // add listeners
            PrologRunner runner = new PrologRunner();
            jip.addEventListener(runner);
                
            JIPTerm query = null;

            jip.consultFile("kermit.pl");
            //jip.consultStream(new StringReader(facts), "facts");

            synchronized(jip) {
                query = jip.getTermParser().parseTerm("?- wouldcallnot(i,it).");
                jip.openQuery(query);
            }

            synchronized(runner) {
                runner.wait();
            }
        } catch (Exception e) {
            e.printStackTrace();            
        }
    }
}
