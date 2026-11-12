#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"
#include "TParameter.h"
#include "TNamed.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace Pythia8;

int main(int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <Ee_GeV> <Ep_GeV> <nEvents> <output.root> [PDFset]\n"
                  << "  [PDFset] is an LHAPDF6 set name for the proton PDF,\n"
                  << "  e.g. NNPDF31_nlo_as_0118 (default), NNPDF31_nnlo_as_0118,\n"
                  << "  NNPDF40_nnlo_as_01180, NNPDF23_nlo_as_0119_qed, etc.\n";
        return 1;
    }

    const double eE      = std::stod(argv[1]);
    const double pE      = std::stod(argv[2]);
    const int    nEvents = std::stoi(argv[3]);
    const std::string outFile = argv[4];
    const std::string pdfSet  = (argc >= 6) ? argv[5] : "CT18NLO";

    // Derive txt output name: replace .root with _sigma.txt
    std::string sigmaFile = outFile;
    size_t dotpos = sigmaFile.rfind(".root");
    if (dotpos != std::string::npos)
        sigmaFile.replace(dotpos, 5, "_sigma.txt");
    else
        sigmaFile += "_sigma.txt";

    const double s_cm = 4.0 * eE * pE;

    // write sigma to the txt file every 1000 events
    const int    LOG_STEP  = 1000;
    const long   LOG_MAX   = 1000000000;

    Pythia pythia;

    pythia.readString("Beams:frameType = 2");
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = -11");
    pythia.readString("Beams:eA = " + std::to_string(pE));
    pythia.readString("Beams:eB = " + std::to_string(eE));
    pythia.readString("WeakBosonExchange:all = on");
    pythia.readString("PhaseSpace:Q2Min = 4.0");
    pythia.readString("PhaseSpace:Q2Max = 10000.0");
    pythia.readString("PhaseSpace:mHatMin = 0.");

    pythia.readString("PDF:pSet = LHAPDF6:" + pdfSet);

    std::cout << "Using proton PDF set: LHAPDF6:" << pdfSet << "\n";

    // Forward-physics tune of Fieg, Kling, Schulz and Sjostrand, PRD 109, 016010 (2024)
    pythia.readString("ColourReconnection:mode = 1");
    pythia.readString("ColourReconnection:allowJunctions = on");

    // beam remnant / diquark fragmentation (forward tune values)
    pythia.readString("BeamRemnants:dampPopcorn = 0");        // dpop: popcorn off
    pythia.readString("BeamRemnants:hardRemnantBaryon = on"); // fremn: on
    pythia.readString("BeamRemnants:aRemnantBaryon = 0.36");  // aremn: fixed to base-tune LSFF value
    pythia.readString("BeamRemnants:bRemnantBaryon = 1.69");  // bremn: fitted

    // primordial kT parameters (forward tune values)
    pythia.readString("BeamRemnants:primordialKTsoft = 0.58");    // sigma_soft (fitted, sigma)
    pythia.readString("BeamRemnants:primordialKThard = 1.8");     // sigma_hard (unchanged)
    pythia.readString("BeamRemnants:halfScaleForKT = 10.0");      // Qhalf
    pythia.readString("BeamRemnants:halfMassForKT = 1.0");        // mhalf (unchanged)
    pythia.readString("BeamRemnants:primordialKTremnant = 0.58"); // sigma_remn = sigma_soft (f=1)

    pythia.init();

    TFile *fout = new TFile(outFile.c_str(), "RECREATE");
    TTree *tree = new TTree("tree", "All DIS observables");

    double Q2, xB, y, W2, s;
    double xL, xpi, t, MX2;
    double pT;
    double En, Pn, px_n, py_n, pz_n;
    double theta_n, eta_n, rapidity_n, phi_n;
    double EePrime, PePrime;
    double theta_e, eta_e, rapidity_e, phi_e;
    double weight;
    int eventNumber;
    int nNeutrons;

    tree->Branch("Q2",          &Q2,          "Q2/D");
    tree->Branch("xB",          &xB,          "xB/D");
    tree->Branch("y",           &y,           "y/D");
    tree->Branch("W2",          &W2,          "W2/D");
    tree->Branch("s",           &s,           "s/D");
    tree->Branch("xL",          &xL,          "xL/D");
    tree->Branch("xpi",         &xpi,         "xpi/D");
    tree->Branch("t",           &t,           "t/D");
    tree->Branch("MX2",         &MX2,         "MX2/D");
    tree->Branch("pT",          &pT,          "pT/D");
    tree->Branch("En",          &En,          "En/D");
    tree->Branch("Pn",          &Pn,          "Pn/D");
    tree->Branch("px_n",        &px_n,        "px_n/D");
    tree->Branch("py_n",        &py_n,        "py_n/D");
    tree->Branch("pz_n",        &pz_n,        "pz_n/D");
    tree->Branch("theta_n",     &theta_n,     "theta_n/D");
    tree->Branch("eta_n",       &eta_n,       "eta_n/D");
    tree->Branch("rapidity_n",  &rapidity_n,  "rapidity_n/D");
    tree->Branch("phi_n",       &phi_n,       "phi_n/D");
    tree->Branch("EePrime",     &EePrime,     "EePrime/D");
    tree->Branch("PePrime",     &PePrime,     "PePrime/D");
    tree->Branch("theta_e",     &theta_e,     "theta_e/D");
    tree->Branch("eta_e",       &eta_e,       "eta_e/D");
    tree->Branch("rapidity_e",  &rapidity_e,  "rapidity_e/D");
    tree->Branch("phi_e",       &phi_e,       "phi_e/D");
    tree->Branch("weight",      &weight,      "weight/D");
    tree->Branch("eventNumber", &eventNumber, "eventNumber/I");
    tree->Branch("nNeutrons",   &nNeutrons,   "nNeutrons/I");

    long nGenerated  = 0;
    long nDIS        = 0;
    long nNeutronEvt = 0;

    // Open sigma log file
    std::ofstream sigLog(sigmaFile);
    sigLog << std::fixed << std::setprecision(6);
    sigLog << "# sigma logged every 1k events up to 2e6, then at final\n";
    sigLog << "# sigma_acc = sigma_total * N_gen / N_acc\n";
    sigLog << std::left
           << std::setw(14) << "# N_gen"
           << std::setw(14) << "N_acc"
           << std::setw(22) << "sigma_total_nb"
           << std::setw(22) << "sigma_acc_nb"
           << "label\n";

    auto logSigma = [&](bool isFinal) {
        double sigGen = pythia.info.sigmaGen() * 1e6; // mb -> nb
        double sigAcc = (nDIS > 0)
                        ? sigGen * static_cast<double>(nGenerated) / static_cast<double>(nDIS)
                        : 0.0;
        std::string label = isFinal ? "FINAL" : "";
        sigLog << std::left
               << std::setw(14) << nGenerated
               << std::setw(14) << nDIS
               << std::setw(22) << sigGen
               << std::setw(22) << sigAcc
               << label << "\n";
        sigLog.flush();
        if (isFinal)
            std::cout << "[SIGMA][FINAL] N_gen=" << nGenerated
                      << "  N_acc=" << nDIS
                      << "  sigma_total=" << sigGen << " nb"
                      << "  sigma_acc="   << sigAcc << " nb\n";
    };

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        nGenerated++;
        if (!pythia.next()) continue;

        Vec4 pProtonIn  = pythia.event[1].p();
        Vec4 pLeptonIn  = pythia.event[2].p();

        Vec4 pLeptonOut;
        bool foundElectron = false;

        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            if (pythia.event[i].id() == -11) {
                pLeptonOut    = pythia.event[i].p();
                foundElectron = true;
                break;
            }
        }
        if (!foundElectron) continue;

        Vec4 q = pLeptonIn - pLeptonOut;

        Q2 = -q.m2Calc();
        if (Q2 <= 0.0) continue;

        double pDotQ = pProtonIn * q;
        if (pDotQ <= 0.0) continue;

        xB = Q2 / (2.0 * pDotQ);
        W2 = (pProtonIn + q).m2Calc();
        y  = Q2 / (s_cm * xB);

        nDIS++;

    // scattered electron
        EePrime   = pLeptonOut.e();
        PePrime   = pLeptonOut.pAbs();
        theta_e   = pLeptonOut.theta();
        eta_e     = pLeptonOut.eta();
        phi_e     = pLeptonOut.phi();
        double denom_e = pLeptonOut.e() - pLeptonOut.pz();
        rapidity_e = (denom_e > 0.0)
                     ? 0.5 * std::log((pLeptonOut.e() + pLeptonOut.pz()) / denom_e)
                     : -999.0;

    // leading neutron
        bool foundNeutron = false;
        double best_xL   = -1.0;
        Vec4   best_pN;
        nNeutrons = 0;

        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            if (pythia.event[i].id() != 2112) continue;

            nNeutrons++;
            Vec4   pN    = pythia.event[i].p();
            double xLtmp = pN.e() / pProtonIn.e();

            if (xLtmp > best_xL) {
                best_xL       = xLtmp;
                best_pN       = pN;
                foundNeutron  = true;
            }
        }

        if (!foundNeutron) continue;
        nNeutronEvt++;

        xL   = best_xL;
        En   = best_pN.e();
        Pn   = best_pN.pAbs();
        px_n = best_pN.px();
        py_n = best_pN.py();
        pz_n = best_pN.pz();
        pT   = best_pN.pT();

        theta_n = best_pN.theta();
        eta_n   = best_pN.eta();
        phi_n   = best_pN.phi();
        double denom_n = best_pN.e() - best_pN.pz();
        rapidity_n = (denom_n > 0.0)
                     ? 0.5 * std::log((best_pN.e() + best_pN.pz()) / denom_n)
                     : -999.0;

        t   = (pProtonIn - best_pN).m2Calc();
        MX2 = (pProtonIn + q - best_pN).m2Calc();

        double oneMinusXL = 1.0 - xL;
        xpi = (oneMinusXL > 0.0) ? xB / oneMinusXL : -1.0;

        weight      = pythia.info.weight();
        eventNumber = iEvent;
        s           = s_cm;

        tree->Fill();

        if (nGenerated <= LOG_MAX && nGenerated % LOG_STEP == 0)
            logSigma(false);
    }

    // Always write final entry
    logSigma(true);

    pythia.stat();

    double sigGen_nb = pythia.info.sigmaGen() * 1e6;
    double sigAcc_nb = (nDIS > 0)
                       ? sigGen_nb * static_cast<double>(nGenerated) / static_cast<double>(nDIS)
                       : 0.0;

    std::cout << "\n=== Final Cross Sections ===\n"
              << "  sigma_total    = " << sigGen_nb << " nb\n"
              << "  sigma_accepted = " << sigAcc_nb << " nb"
              << "  (N_gen=" << nGenerated << ", N_acc=nDIS=" << nDIS << ")\n";

    TParameter<double>("sigmaGen_nb",  sigGen_nb).Write();
    TParameter<double>("sigmaAcc_nb",  sigAcc_nb).Write();
    TParameter<double>("weightSum",    pythia.info.weightSum()).Write();
    TParameter<double>("nGenerated",   nGenerated).Write();
    TParameter<double>("nDIS",         nDIS).Write();
    TParameter<double>("nNeutronEvt",  nNeutronEvt).Write();
    TNamed("pdfSet", ("LHAPDF6:" + pdfSet).c_str()).Write();
    TNamed("forwardTune", "Fieg-Kling-Schulz-Sjostrand PRD 109 016010 (2024), Table I, QCDCR-based tune").Write();

    tree->Write();
    fout->Close();
    sigLog.close();

    return 0;
}