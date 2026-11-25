import StrawberrySensorInsights from "./StrawberrySensorInsights";
import NitrogenResultInsights from "./NitrogenResultInsights";
import PistachioSpectralInsights from "./PistachioSpectralInsights";
import TomatoSpectralInsights from "./TomatoSpectralInsights";
import "./FoodGuidance.css";
import { useEffect, useState } from "react";
import "./App.css";
import { authenticatedFetch } from "./authSession";
import RgbMoldResearchReport from "./RgbMoldResearchReport.jsx";
import PapayaSpectralInsights from "./PapayaSpectralInsights";

const API_URL = "http://127.0.0.1:8000";
const HSI_FOODS = ["pistachio", "tomato", "papaya", "eggplant", "cabbage"];

export default function App({ initialModule = "hsi", initialRgbMode = "freshness" }) {
  const [activeModule, setActiveModule] = useState(initialModule);
  const [rgbMode, setRgbMode] = useState(initialRgbMode);
  const [hsiMode, setHsiMode] = useState("pistachio");
  const [nitrogenLoading, setNitrogenLoading] = useState(false);
  const [nitrogenResult, setNitrogenResult] = useState(null);
  const [nitrogenError, setNitrogenError] = useState("");

  // Eggplant/Cabbage user-upload workflow
  const [nitrogenMode, setNitrogenMode] = useState("upload");
  const [nitrogenHdr, setNitrogenHdr] = useState(null);
  const [nitrogenBil, setNitrogenBil] = useState(null);
  const [nitrogenUploadLoading, setNitrogenUploadLoading] = useState(false);
  const [nitrogenUploadResult, setNitrogenUploadResult] = useState(null);
  const [nitrogenUploadError, setNitrogenUploadError] = useState("");

  // Pistachio HSI state
  const [hdr, setHdr] = useState(null);
  const [bil, setBil] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // RGB state shared by mutually exclusive workflows
  const [rgbImage, setRgbImage] = useState(null);
  const [rgbPreview, setRgbPreview] = useState("");
  const [rgbLoading, setRgbLoading] = useState(false);
  const [rgbResult, setRgbResult] = useState(null);
  const [rgbError, setRgbError] = useState("");

  const [sensorFile, setSensorFile] = useState(null);
  const [sensorLoading, setSensorLoading] = useState(false);
  const [sensorResult, setSensorResult] = useState(null);
  const [sensorError, setSensorError] = useState("");

  async function analyzeSensor(event) {
    event.preventDefault();

    if (!sensorFile) return;

    setSensorLoading(true);
    setSensorError("");
    setSensorResult(null);

    try {
      const formData = new FormData();
      formData.append("file", sensorFile);

      const response = await authenticatedFetch(`${API_URL}/predict/sensor/strawberry`,
        {
          method: "POST",
          body: formData
        }
      );

      const data = await response.json();

      if (!response.ok) {
        const detail = data.detail;
        const message =
          typeof detail === "string"
            ? detail
            : detail?.message
              ? `${detail.message} ${detail.missing_columns?.slice(0, 8).join(", ") || ""}`
              : "Strawberry sensor analysis failed.";

        throw new Error(message);
      }

      setSensorResult(data);
    } catch (error) {
      setSensorError(error.message || "Unable to analyze CSV.");
    } finally {
      setSensorLoading(false);
    }
  }



  // Papaya exploratory HSI analysis
  const [papayaHdr, setPapayaHdr] = useState(null);
  const [papayaImg, setPapayaImg] = useState(null);
  const [papayaLoading, setPapayaLoading] = useState(false);
  const [papayaResult, setPapayaResult] = useState(null);
  const [papayaError, setPapayaError] = useState("");

  async function analyzePapaya(event) {
    event.preventDefault();

    if (!papayaHdr || !papayaImg) return;

    setPapayaLoading(true);
    setPapayaError("");
    setPapayaResult(null);

    try {
      const formData = new FormData();
      formData.append("hdr", papayaHdr);
      formData.append("img", papayaImg);

      const response = await authenticatedFetch(`${API_URL}/predict/hsi/papaya/spectral`,
        { method: "POST", body: formData }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Papaya HSI analysis failed."
        );
      }

      setPapayaResult(data);
    } catch (error) {
      setPapayaError(error.message || "Unable to analyze Papaya HSI.");
    } finally {
      setPapayaLoading(false);
    }
  }


  // Tomato MATLAB hyperspectral cube analysis
  const [tomatoMatFile, setTomatoMatFile] = useState(null);
  const [tomatoMatLoading, setTomatoMatLoading] = useState(false);
  const [tomatoMatResult, setTomatoMatResult] = useState(null);
  const [tomatoMatError, setTomatoMatError] = useState("");

  async function analyzeTomatoMat(event) {
    event.preventDefault();

    if (!tomatoMatFile) return;

    const maxBytes = 160 * 1024 * 1024;

    if (tomatoMatFile.size > maxBytes) {
      setTomatoMatError("MAT file exceeds the 160 MiB limit.");
      return;
    }

    setTomatoMatLoading(true);
    setTomatoMatError("");
    setTomatoMatResult(null);

    try {
      const formData = new FormData();
      formData.append("file", tomatoMatFile);

      const response = await authenticatedFetch(`${API_URL}/predict/hsi/tomato/mat`,
        { method: "POST", body: formData }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Tomato MATLAB analysis failed."
        );
      }

      setTomatoMatResult(data);
    } catch (error) {
      setTomatoMatError(
        error.message || "Unable to analyze Tomato MATLAB cube."
      );
    } finally {
      setTomatoMatLoading(false);
    }
  }

  const [tomatoFile, setTomatoFile] = useState(null);
  const [tomatoLoading, setTomatoLoading] = useState(false);
  const [tomatoResult, setTomatoResult] = useState(null);
  const [tomatoError, setTomatoError] = useState("");

  async function analyzeTomatoSpectral(event) {
    event.preventDefault();
    if (!tomatoFile) return;

    setTomatoLoading(true);
    setTomatoError("");
    setTomatoResult(null);

    try {
      const formData = new FormData();
      formData.append("file", tomatoFile);

      const response = await authenticatedFetch(`${API_URL}/predict/hsi/tomato/spectral`,
        { method: "POST", body: formData }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Tomato spectral analysis failed."
        );
      }

      setTomatoResult(data);
    } catch (error) {
      setTomatoError(error.message || "Unable to analyze Tomato CSV.");
    } finally {
      setTomatoLoading(false);
    }
  }

  useEffect(() => {
    if (!rgbImage) {
      setRgbPreview("");
      return;
    }
    const url = URL.createObjectURL(rgbImage);
    setRgbPreview(url);
  
  return () => URL.revokeObjectURL(url);
  }, [rgbImage]);

  function changeRgbMode(nextMode) {
    if (rgbLoading || nextMode === rgbMode) return;
    setRgbMode(nextMode);
    setRgbImage(null);
    setRgbResult(null);
    setRgbError("");
  }

  function changeHsiMode(nextMode) {
    if (loading || nitrogenLoading || nitrogenUploadLoading || nextMode === hsiMode) return;
    setHsiMode(nextMode);
    setNitrogenResult(null);
    setNitrogenError("");
    setNitrogenHdr(null);
    setNitrogenBil(null);
    setNitrogenUploadResult(null);
    setNitrogenUploadError("");
    setResult(null);
    setError("");
  }

  async function analyze(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!hdr || !bil) {
      setError("Please select both HDR and BIL files.");
      return;
    }

    const hdrStem = hdr.name.replace(/\.hdr$/i, "");
    const bilStem = bil.name.replace(/\.bil$/i, "");
    if (hdrStem.toLowerCase() !== bilStem.toLowerCase()) {
      setError("HDR and BIL files must belong to the same cube.");
      return;
    }

    const formData = new FormData();
    formData.append("hdr", hdr);
    formData.append("bil", bil);
    setLoading(true);

    try {
      const response = await authenticatedFetch(`${API_URL}/predict/hsi/pistachio`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Analysis failed (HTTP ${response.status})`
        );
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to complete analysis.");
    } finally {
      setLoading(false);
    }
  }

  async function analyzeNitrogen(food) {
    if (nitrogenLoading) return;
    if (!["eggplant", "cabbage"].includes(food)) {
      setNitrogenError("Unsupported HSI dataset.");
      return;
    }

    setNitrogenLoading(true);
    setNitrogenError("");
    setNitrogenResult(null);

    try {
      const response = await authenticatedFetch(`${API_URL}/predict/hsi/${food}/local`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Analysis failed (HTTP ${response.status})`
        );
      }
      setNitrogenResult(data);
    } catch (err) {
      setNitrogenError(err.message || "Unable to complete HSI analysis.");
    } finally {
      setNitrogenLoading(false);
    }
  }

  async function analyzeNitrogenUpload(event) {
    event.preventDefault();
    setNitrogenUploadError("");
    setNitrogenUploadResult(null);

    if (!nitrogenHdr || !nitrogenBil) {
      setNitrogenUploadError("Please select both HDR and spectral binary files.");
      return;
    }
    if (!/\.hdr$/i.test(nitrogenHdr.name)) {
      setNitrogenUploadError("Metadata file must have the .hdr extension.");
      return;
    }
    if (!/\.bil$/i.test(nitrogenBil.name) && nitrogenBil.name.includes(".")) {
      setNitrogenUploadError("Spectral data must be .bil or an extensionless ENVI binary.");
      return;
    }
    if (nitrogenHdr.size > 2 * 1024 * 1024 || nitrogenBil.size > 100 * 1024 * 1024) {
      setNitrogenUploadError("Maximum sizes: HDR 2 MB; spectral binary 100 MB.");
      return;
    }
    const hdrStem = nitrogenHdr.name.replace(/\.hdr$/i, "").toLowerCase();
    const binaryStem = nitrogenBil.name.replace(/\.bil$/i, "").toLowerCase();
    if (hdrStem !== binaryStem) {
      setNitrogenUploadError("Choose an HDR and spectral binary with matching filenames.");
      return;
    }

    const formData = new FormData();
    formData.append("hdr", nitrogenHdr);
    formData.append("bil", nitrogenBil);
    setNitrogenUploadLoading(true);
    try {
      const response = await authenticatedFetch(`${API_URL}/predict/hsi/${hsiMode}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(typeof data.detail === "string"
          ? data.detail : `Analysis failed (HTTP ${response.status})`);
      }
      setNitrogenUploadResult(data);
    } catch (err) {
      setNitrogenUploadError(err.message || "Unable to analyze the uploaded cube.");
    } finally {
      setNitrogenUploadLoading(false);
    }
  }

  async function analyzeRgb(event) {
    event.preventDefault();
    setRgbError("");
    setRgbResult(null);

    if (!rgbImage) {
      setRgbError("Please select an RGB image.");
      return;
    }
    if (rgbImage.size > 10 * 1024 * 1024) {
      setRgbError("Image must be 10 MB or smaller.");
      return;
    }

    const formData = new FormData();
    formData.append("image", rgbImage);
    setRgbLoading(true);

    try {
      const endpoint =
        rgbMode === "freshness"
          ? "/predict/rgb/freshness"
          : "/predict/rgb/mold";
      const response = await authenticatedFetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Analysis failed (HTTP ${response.status})`
        );
      }
      setRgbResult(data);
    } catch (err) {
      setRgbError(err.message || "Unable to complete RGB analysis.");
    } finally {
      setRgbLoading(false);
    }
  }

  const prediction = result?.result;
  const below = prediction?.prediction === "Below_Threshold";
  const rgbPrediction = rgbResult?.result;
  const rgbGuidance = rgbResult?.guidance;
  const isFreshness = rgbMode === "freshness";
  const freshnessIsFresh = rgbPrediction?.freshness === "Fresh";
  const moldDetected = rgbPrediction?.prediction === "Mold";

  return (
    <div className="app">
      <header className="navbar">
        <div className="brand">
          <span className="brand-icon">✦</span>
          SpectraGuard
        </div>
        <span className="nav-tag">AI Food Quality Platform</span>
      </header>

      <main className="container">
        <section className="hero">
          <div className="eyebrow">INTELLIGENT FOOD ANALYSIS</div>
          <h1>Understand food quality through AI.</h1>
          <p>
            A unified research platform combining RGB image analysis,
            hyperspectral imaging and multispectral sensing.
          </p>
        </section>

        <section className="module-grid">
          <article
            className={`module-card ${activeModule === "rgb" ? "selected" : ""}`}
            onClick={() => setActiveModule("rgb")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                setActiveModule("rgb");
              }
            }}
          >
            <div className="module-icon">◉</div>
            <h3>RGB Analysis</h3>
            <p>
              Freshness classification from food photographs and experimental
              mold detection from microscope-assisted images.
            </p>
            <span className="status active">Freshness + mold active</span>
          </article>

          <article
            className={`module-card ${activeModule === "hsi" ? "selected" : ""}`}
            onClick={() => setActiveModule("hsi")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                setActiveModule("hsi");
              }
            }}
          >
            <div className="module-icon">▦</div>
            <h3>Hyperspectral Analysis</h3>
            <p>Analyze spectral cubes using trained ML models.</p>
            <span className="status active">Pistachio + Eggplant/Cabbage uploads active</span>
          </article>

          
          <article
            className={`module-card ${activeModule === "sensor" ? "selected" : ""}`}
            onClick={() => setActiveModule("sensor")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                setActiveModule("sensor");
              }
            }}
          >
            <div className="module-icon">?</div>
            <h3>Multispectral Sensor Analysis</h3>
            <p>Explore recorded Strawberry sensor measurements.</p>
            <span className="status active">CSV upload + dataset analysis</span>
          </article>
        </section>

        {activeModule === "rgb" && (
          <>
            <section className="workspace">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">MODULE A</div>
                  <h2>RGB Food Analysis</h2>
                  <p>Choose the type of analysis before uploading an image.</p>
                </div>
                <span className="version">Experimental v0.2</span>
              </div>

              <div className="rgb-mode-selector" role="group" aria-label="RGB analysis type">
                <button
                  type="button"
                  className={`rgb-mode-button ${isFreshness ? "active" : ""}`}
                  aria-pressed={isFreshness}
                  disabled={rgbLoading}
                  onClick={() => changeRgbMode("freshness")}
                >
                  Freshness Analysis
                </button>
                <button
                  type="button"
                  className={`rgb-mode-button ${!isFreshness ? "active" : ""}`}
                  aria-pressed={!isFreshness}
                  disabled={rgbLoading}
                  onClick={() => changeRgbMode("mold")}
                >
                  Mold Detection
                </button>
              </div>

              <p className="loading-note">
                {isFreshness
                  ? "Upload an ordinary RGB photograph of Apple, Banana, Strawberry or Tomato. The model predicts food type and Fresh/Rotten class."
                  : "Upload a microscope-assisted RGB photograph of Tomato, Orange, Onion or Carrot. Ordinary phone photographs are not validated for this mold model."}
              </p>

              <form className="upload-form" onSubmit={analyzeRgb}>
                <div className="upload-grid">
                  <label className="upload-box">
                    <span className="upload-symbol">↑</span>
                    <strong>RGB Food Image</strong>
                    <span>{rgbImage ? rgbImage.name : "Select JPG, JPEG or PNG"}</span>
                    <input
                      key={rgbMode}
                      type="file"
                      accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                      onChange={(e) => {
                        setRgbImage(e.target.files?.[0] || null);
                        setRgbResult(null);
                        setRgbError("");
                      }}
                    />
                  </label>

                  {rgbPreview && (
                    <div className="rgb-preview-box">
                      <img
                        src={rgbPreview}
                        alt="Selected RGB food sample"
                        className="rgb-preview-image"
                      />
                    </div>
                  )}
                </div>

                <p className="loading-note">Maximum image size: 10 MB.</p>

                <button
                  className="analyze-button"
                  type="submit"
                  disabled={rgbLoading || !rgbImage}
                >
                  {rgbLoading
                    ? "Analyzing RGB image..."
                    : isFreshness
                      ? "Analyze Freshness →"
                      : "Detect Mold →"}
                </button>

                {rgbLoading && (
                  <p className="loading-note">
                    Running {isFreshness ? "MobileNet freshness" : "MobileNetV2 mold"} inference.
                    Please wait.
                  </p>
                )}
                {rgbError && <div className="error-message">{rgbError}</div>}
              </form>
            </section>

            {rgbPrediction && (
              <section className="results">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">ANALYSIS COMPLETE</div>
                    <h2>
                      {isFreshness
                        ? "RGB Freshness Analysis Results"
                        : "RGB Mold Detection Results"}
                    </h2>
                  </div>
                  <span className="status active">Completed</span>
                </div>

                {isFreshness ? (
                  <>
                    <div className="prediction-card">
                      <span>Experimental eight-class MobileNet classification</span>
                      <h2 className={freshnessIsFresh ? "below" : "above"}>
                        {rgbPrediction.food} — {rgbPrediction.freshness}
                      </h2>
                      <p>Predicted category: {rgbPrediction.prediction}</p>
                    </div>
                    <div className="metrics">
                      <div className="metric">
                        <span>Food identified</span>
                        <strong>{rgbPrediction.food}</strong>
                      </div>
                      <div className="metric">
                        <span>Freshness class</span>
                        <strong>{rgbPrediction.freshness}</strong>
                      </div>
                      <div className="metric">
                        <span>Top-class model score</span>
                        <strong>
                          {(rgbPrediction.model_score * 100).toFixed(2)}%
                        </strong>
                        <small>Uncalibrated softmax score</small>
                      </div>
                      <div className="metric">
                        <span>Model</span>
                        <strong>MobileNet</strong>
                      </div>
                    </div>

                    {rgbGuidance?.available && (
                      <div className="food-guidance">
                        <h3>Food Storage &amp; Spoilage Guidance</h3>

                        <div className="food-guidance-card">
                          <h4>Recommended Action</h4>
                          <p>{rgbGuidance.recommendation}</p>
                        </div>

                        <div className="food-guidance-grid">
                          <div className="food-guidance-card">
                            <h4>Storage Recommendations</h4>
                            <ul>
                              {(rgbGuidance.storage_tips || []).map(
                                (tip, index) => (
                                  <li key={index}>{tip}</li>
                                )
                              )}
                            </ul>
                          </div>

                          <div className="food-guidance-card">
                            <h4>Visible Spoilage Signs</h4>
                            <p>
                              Inspect the actual food for:
                            </p>
                            <ul>
                              {(rgbGuidance.visible_spoilage_signs || []).map(
                                (sign, index) => (
                                  <li key={index}>{sign}</li>
                                )
                              )}
                            </ul>
                          </div>
                        </div>

                        <p className="food-guidance-disclaimer">
                          {rgbGuidance.disclaimer}
                        </p>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="prediction-card">
                      <span>Experimental MobileNetV2 classification</span>
                      <h2 className={moldDetected ? "above" : "below"}>
                        {rgbPrediction.prediction}
                      </h2>
                      <p>Classification threshold: {rgbPrediction.threshold}</p>
                    </div>
                    <div className="metrics">
                      <div className="metric">
                        <span>Mold model score</span>
                        <strong>
                          {(rgbPrediction.mold_score * 100).toFixed(2)}%
                        </strong>
                        <small>Uncalibrated sigmoid score</small>
                      </div>
                      <div className="metric">
                        <span>Predicted class</span>
                        <strong>{rgbPrediction.prediction}</strong>
                      </div>
                      <div className="metric">
                        <span>Model</span>
                        <strong>MobileNetV2</strong>
                      </div>
                      <div className="metric">
                        <span>Input type</span>
                        <strong>Microscope-assisted RGB</strong>
                      </div>
                    </div>
                  </>
                )}
                {!isFreshness && rgbResult?.research_report && (
                  <RgbMoldResearchReport
                    report={rgbResult.research_report}
                  />
                )}

                <div className="warning">
                  {rgbResult.warning || rgbPrediction.warning}
                </div>
              </section>
            )}
          </>
        )}

        {activeModule === "hsi" && (
          <>
            <section className="workspace">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">HYPERSPECTRAL ANALYSIS</div>
                  <h2>Select HSI Dataset</h2>
                  <p>Choose a supported hyperspectral analysis workflow.</p>
                </div>
              </div>

              <div
                className="rgb-mode-selector"
                role="group"
                aria-label="HSI dataset selection"
              >
                {HSI_FOODS.map((food) => (
                  <button
                    key={food}
                    type="button"
                    className={`rgb-mode-button ${hsiMode === food ? "active" : ""}`}
                    aria-pressed={hsiMode === food}
                    disabled={loading || nitrogenLoading || nitrogenUploadLoading}
                    onClick={() => changeHsiMode(food)}
                  >
                    {food.charAt(0).toUpperCase() + food.slice(1)}
                  </button>
                ))}
              </div>
            </section>



            {hsiMode === "papaya" && (
              <>
                <section className="workspace">
                  <div className="section-heading">
                    <div>
                      <div className="eyebrow">MODULE B - HYPERSPECTRAL RESEARCH</div>
                      <h2>Papaya Hyperspectral Analysis</h2>
                      <p>
                        Upload a matching ENVI HDR and IMG pair to explore
                        its 150-band spectral signature and PCA structure.
                      </p>
                    </div>
                    <span className="version">Research Analysis</span>
                  </div>

                  <form className="upload-form" onSubmit={analyzePapaya}>
                    <div className="upload-grid">
                      <label className="upload-box">
                        <span className="upload-symbol">1</span>
                        <strong>ENVI Header</strong>
                        <span>
                          {papayaHdr ? papayaHdr.name : "Select .hdr file"}
                        </span>
                        <input
                          type="file"
                          accept=".hdr"
                          onChange={(event) => {
                            setPapayaHdr(event.target.files?.[0] || null);
                            setPapayaResult(null);
                            setPapayaError("");
                          }}
                        />
                      </label>

                      <label className="upload-box">
                        <span className="upload-symbol">2</span>
                        <strong>Hyperspectral Cube</strong>
                        <span>
                          {papayaImg ? papayaImg.name : "Select .img file"}
                        </span>
                        <input
                          type="file"
                          accept=".img"
                          onChange={(event) => {
                            setPapayaImg(event.target.files?.[0] || null);
                            setPapayaResult(null);
                            setPapayaError("");
                          }}
                        />
                      </label>
                    </div>

                    <button
                      type="submit"
                      className="analyze-button"
                      disabled={!papayaHdr || !papayaImg || papayaLoading}
                    >
                      {papayaLoading
                        ? "Analyzing Papaya cube..."
                        : "Analyze Papaya HSI"}
                    </button>

                    {papayaLoading && (
                      <p className="loading-note">
                        Extracting spectral signatures and computing PCA.
                      </p>
                    )}

                    {papayaError && (
                      <div className="error-message">{papayaError}</div>
                    )}
                  </form>
                </section>

                {papayaResult && (
                  <section className="results">
                    <div className="section-heading">
                      <div>
                        <div className="eyebrow">HYPERSPECTRAL RESULTS</div>
                        <h2>Papaya Spectral Analysis</h2>
                      </div>
                      <span className="status active">Completed</span>
                    </div>

                    <div className="prediction-card">
                      <h3>Cube Information</h3>
                      <p>File: {papayaResult.filename}</p>
                      <p>
                        Dimensions: {papayaResult.shape.join(" x ")}
                      </p>
                      <p>Pixels: {papayaResult.total_pixels}</p>
                      <p>Spectral bands: {papayaResult.spectral_bands}</p>
                      <p>
                        Mean signal: {papayaResult.signal_mean.toFixed(6)}
                      </p>
                      <p>
                        Signal range: {papayaResult.signal_min.toFixed(6)}
                        {" to "}
                        {papayaResult.signal_max.toFixed(6)}
                      </p>
                    </div>

                    <div className="prediction-card">
                      <h3>Mean Spectral Signature</h3>
                      <p>
                        Average signal across all 1,024 pixels,
                        plotted by spectral band index.
                      </p>

                      <svg
                        viewBox="0 0 720 280"
                        role="img"
                        aria-label="Papaya mean spectral signature"
                        style={{
                          width: "100%",
                          height: "auto",
                          marginTop: "16px"
                        }}
                      >
                        <line
                          x1="55" y1="235" x2="690" y2="235"
                          stroke="currentColor" opacity="0.5"
                        />
                        <line
                          x1="55" y1="20" x2="55" y2="235"
                          stroke="currentColor" opacity="0.5"
                        />
                        <polyline
                          fill="none"
                          stroke="#2b86d1"
                          strokeWidth="2.5"
                          strokeLinejoin="round"
                          points={(() => {
                            const values = papayaResult.mean_spectrum.map(
                              (point) => point.mean
                            );
                            const min = Math.min(...values);
                            const max = Math.max(...values);
                            const range = max - min || 1;

                            return papayaResult.mean_spectrum.map(
                              (point, index) => {
                                const x = 55 + index * 635 /
                                  Math.max(values.length - 1, 1);
                                const y = 225 -
                                  (point.mean - min) / range * 195;
                                return `${x.toFixed(2)},${y.toFixed(2)}`;
                              }
                            ).join(" ");
                          })()}
                        />
                        <text x="55" y="255" fontSize="13"
                          fill="currentColor">Band 1</text>
                        <text x="615" y="255" fontSize="13"
                          fill="currentColor">Band 150</text>
                        <text x="320" y="275" fontSize="13"
                          fill="currentColor">Spectral band index</text>
                      </svg>
                      <PapayaSpectralInsights
                        result={papayaResult}
                        type="spectrum"
                      />
                    </div>

                    <div className="prediction-card">
                      <h3>PCA Explained Variance</h3>
                      {papayaResult.pca_explained_variance_percent.map(
                        (value, index) => (
                          <p key={index}>
                            PC{index + 1}: {value.toFixed(2)}%
                          </p>
                        )
                      )}
                      <p>
                        First three components combined:{" "}
                        {papayaResult.pca_explained_variance_percent
                          .reduce((sum, value) => sum + value, 0)
                          .toFixed(2)}%
                      </p>
                      <PapayaSpectralInsights
                        result={papayaResult}
                        type="pca"
                      />
                    </div>

                    <div className="warning">
                      {papayaResult.warning}
                    </div>
                  </section>
                )}
              </>
            )}

            {hsiMode === "tomato" && (
              <section className="workspace">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">MODULE B ? SPECTRAL RESEARCH</div>
                    <h2>Tomato Hyperspectral Analysis</h2>
                    <p>
                      Exploratory spectral analysis of Healthy tomato samples
                      and four fungal-associated classes.
                    </p>
                  </div>
                  <span className="version">Research Analysis</span>
                </div>

                <div className="prediction-card">
                  <span>Five-class hyperspectral study</span>
                  <h2>Spectral Signature Analysis</h2>
                  <p>
                    Classes: Healthy, Alternaria alternata,
                    Alternaria solani, Botrytis cinerea,
                    and Fusarium oxysporum.
                  </p>
                  <p>
                    These are dataset-level spectral analyses,
                    not a validated fungal-species prediction service.
                  </p>
                </div>

                <div className="metrics">
                  <div className="metric">
                    <span>Classes analyzed</span>
                    <strong>5</strong>
                  </div>
                  <div className="metric">
                    <span>Analysis</span>
                    <strong>Spectral + PCA</strong>
                  </div>
                  <div className="metric">
                    <span>Classification</span>
                    <strong>Exploratory</strong>
                  </div>
                </div>


                <div className="prediction-card">
                  <h3>Upload Tomato Spectral Signature</h3>
                  <p>
                    Upload a CSV containing 550 wavelength measurements
                    with wavelength_nm and mean or mean_raw_response columns.
                    The graph compares normalized curve shapes with one
                    Healthy reference sample.
                  </p>
                </div>

                <form className="upload-form" onSubmit={analyzeTomatoSpectral}>
                  <div className="upload-grid">
                    <label className="upload-box">
                      <span className="upload-symbol">?</span>
                      <strong>Tomato Spectral CSV</strong>
                      <span>
                        {tomatoFile ? tomatoFile.name : "Select .csv file"}
                      </span>
                      <input
                        type="file"
                        accept=".csv,text/csv"
                        disabled={tomatoLoading}
                        onChange={(event) => {
                          setTomatoFile(event.target.files?.[0] || null);
                          setTomatoResult(null);
                          setTomatoError("");
                        }}
                      />
                    </label>
                  </div>

                  <p className="loading-note">
                    Maximum file size: 2 MB. Requires 550 spectral bands
                    on the Tomato reference wavelength grid.
                  </p>

                  <button
                    className="analyze-button"
                    type="submit"
                    disabled={!tomatoFile || tomatoLoading}
                  >
                    {tomatoLoading
                      ? "Analyzing spectral signature..."
                      : "Analyze Tomato Spectrum ?"}
                  </button>

                  {tomatoError && (
                    <div className="error-message">{tomatoError}</div>
                  )}
                </form>

                {tomatoResult && (
                  <section className="results">
                    <div className="section-heading">
                      <div>
                        <div className="eyebrow">SPECTRAL ANALYSIS COMPLETE</div>
                        <h2>Uploaded Tomato Spectrum</h2>
                        <p>{tomatoResult.filename}</p>
                      </div>
                      <span className="status active">Completed</span>
                    </div>

                    <div className="metrics">
                      <div className="metric">
                        <span>Spectral bands</span>
                        <strong>{tomatoResult.bands}</strong>
                      </div>
                      <div className="metric">
                        <span>Mean signal</span>
                        <strong>{tomatoResult.signal_mean.toFixed(4)}</strong>
                      </div>
                      <div className="metric">
                        <span>Peak wavelength</span>
                        <strong>
                          {tomatoResult.peak_wavelength_nm.toFixed(2)} nm
                        </strong>
                      </div>
                      <div className="metric">
                        <span>Healthy reference correlation</span>
                        <strong>
                          {tomatoResult.reference_correlation === null
                            ? "N/A"
                            : tomatoResult.reference_correlation.toFixed(4)}
                        </strong>
                        <small>Similarity only; not a diagnosis</small>
                      </div>
                    </div>

                    <div className="prediction-card">
                      <h3>Normalized Spectral Curve Comparison</h3>
                      <p>
                        Blue: uploaded sample. Orange: Healthy reference.
                        Curves are independently min-max normalized.
                      </p>

                      <div style={{overflowX: "auto"}}>
                        <svg
                          viewBox="0 0 760 330"
                          role="img"
                          aria-label="Tomato uploaded and reference spectral curves"
                          style={{
                            width: "100%",
                            minWidth: "520px",
                            background: "#ffffff",
                            borderRadius: "12px"
                          }}
                        >
                          {[0, 0.25, 0.5, 0.75, 1].map((value) => (
                            <g key={value}>
                              <line
                                x1="65" x2="735"
                                y1={280 - value * 240}
                                y2={280 - value * 240}
                                stroke="#e5e7eb"
                              />
                              <text
                                x="52"
                                y={284 - value * 240}
                                fontSize="12"
                                fill="#475569"
                                textAnchor="end"
                              >
                                {value.toFixed(2)}
                              </text>
                            </g>
                          ))}

                          {[400, 500, 600, 700, 800, 900, 950].map((w) => (
                            <text
                              key={w}
                              x={65 + (w - 400) / 550 * 670}
                              y="305"
                              fontSize="12"
                              fill="#475569"
                              textAnchor="middle"
                            >
                              {w}
                            </text>
                          ))}

                          <polyline
                            fill="none"
                            stroke="#2563eb"
                            strokeWidth="2"
                            points={tomatoResult.curve.map((p) =>
                              `${65 + (p.wavelength_nm - 400) / 550 * 670},${280 - p.normalized_signal * 240}`
                            ).join(" ")}
                          />

                          <polyline
                            fill="none"
                            stroke="#f97316"
                            strokeWidth="2"
                            strokeDasharray="6 4"
                            points={tomatoResult.curve.map((p) =>
                              `${65 + (p.wavelength_nm - 400) / 550 * 670},${280 - p.normalized_reference * 240}`
                            ).join(" ")}
                          />

                          <text
                            x="400" y="326"
                            fontSize="13"
                            fill="#334155"
                            textAnchor="middle"
                          >
                            Wavelength (nm)
                          </text>
                        </svg>
                      </div>

                      <p style={{marginTop: "12px"}}>
                        <span style={{color: "#2563eb"}}>??</span>
                        {" "}Uploaded sample{"  "}
                        <span style={{color: "#f97316"}}>??</span>
                        {" "}Healthy reference
                      </p>
                    </div>

                    <TomatoSpectralInsights
                      type="csv"
                      result={tomatoResult}
                    />

                    <div className="warning">
                      {tomatoResult.warning}
                    </div>
                  </section>
                )}


                <div className="prediction-card">
                  <h3>Upload Tomato MATLAB Hyperspectral Cube</h3>
                  <p>
                    Analyze an original 140 x 280 x 550 Tomato HSI cube
                    stored in MATLAB 5.0 .mat format.
                    The analysis extracts a normalized mean spectral
                    signature without performing fungal classification.
                  </p>
                </div>

                <form className="upload-form" onSubmit={analyzeTomatoMat}>
                  <div className="upload-grid">
                    <label className="upload-box">
                      <span className="upload-symbol">MAT</span>
                      <strong>MATLAB Hyperspectral Cube</strong>
                      <span>
                        {tomatoMatFile
                          ? tomatoMatFile.name
                          : "Select .mat file"}
                      </span>
                      <input
                        type="file"
                        accept=".mat"
                        disabled={tomatoMatLoading}
                        onChange={(event) => {
                          setTomatoMatFile(event.target.files?.[0] || null);
                          setTomatoMatResult(null);
                          setTomatoMatError("");
                        }}
                      />
                    </label>
                  </div>

                  <p className="loading-note">
                    Maximum file size: 160 MiB.
                    Expected cube: 140 x 280 x 550, MATLAB 5.0 format.
                  </p>

                  <button
                    className="analyze-button"
                    type="submit"
                    disabled={!tomatoMatFile || tomatoMatLoading}
                  >
                    {tomatoMatLoading
                      ? "Analyzing MATLAB cube..."
                      : "Analyze Tomato MATLAB Cube"}
                  </button>

                  {tomatoMatLoading && (
                    <p className="loading-note">
                      Uploading and processing the hyperspectral cube.
                      Large files may take additional time.
                    </p>
                  )}

                  {tomatoMatError && (
                    <div className="error-message">
                      {tomatoMatError}
                    </div>
                  )}
                </form>

                {tomatoMatResult && (
                  <section className="results">
                    <div className="section-heading">
                      <div>
                        <div className="eyebrow">MATLAB CUBE ANALYSIS COMPLETE</div>
                        <h2>Tomato HSI Cube Results</h2>
                        <p>{tomatoMatResult.filename}</p>
                      </div>
                      <span className="status active">Completed</span>
                    </div>

                    <div className="metrics">
                      <div className="metric">
                        <span>Cube dimensions</span>
                        <strong>{tomatoMatResult.shape.join(" x ")}</strong>
                      </div>
                      <div className="metric">
                        <span>Pixels</span>
                        <strong>
                          {tomatoMatResult.total_pixels.toLocaleString()}
                        </strong>
                      </div>
                      <div className="metric">
                        <span>Spectral bands</span>
                        <strong>{tomatoMatResult.spectral_bands}</strong>
                      </div>
                      <div className="metric">
                        <span>Raw mean signal</span>
                        <strong>
                          {tomatoMatResult.raw_mean.toFixed(6)}
                        </strong>
                      </div>
                    </div>

                    <div className="prediction-card">
                      <h3>Cube Signal Statistics</h3>
                      <p>
                        MATLAB variable: {tomatoMatResult.variable_name}
                      </p>
                      <p>
                        Raw minimum: {tomatoMatResult.raw_min.toFixed(6)}
                      </p>
                      <p>
                        Raw maximum: {tomatoMatResult.raw_max.toFixed(6)}
                      </p>
                      <p>
                        Normalization: {tomatoMatResult.normalization}
                      </p>
                    </div>

                    <div className="prediction-card">
                      <h3>Normalized Mean Spectral Signature</h3>
                      <p>
                        Mean spectral signal across 39,200 pixels.
                        Spectral positions are band indices,
                        not calibrated wavelengths.
                      </p>

                      <div style={{overflowX: "auto"}}>
                        <svg
                          viewBox="0 0 760 330"
                          role="img"
                          aria-label="Tomato MATLAB cube mean spectral signature"
                          style={{
                            width: "100%",
                            minWidth: "520px",
                            background: "#ffffff",
                            borderRadius: "12px"
                          }}
                        >
                          {[0, 0.25, 0.5, 0.75, 1].map((value) => (
                            <g key={value}>
                              <line
                                x1="65"
                                x2="735"
                                y1={280 - value * 240}
                                y2={280 - value * 240}
                                stroke="#e5e7eb"
                              />
                              <text
                                x="52"
                                y={284 - value * 240}
                                fontSize="12"
                                fill="#475569"
                                textAnchor="end"
                              >
                                {value.toFixed(2)}
                              </text>
                            </g>
                          ))}

                          {[1, 100, 200, 300, 400, 500, 550].map(
                            (band) => (
                              <text
                                key={band}
                                x={65 + (band - 1) / 549 * 670}
                                y="305"
                                fontSize="12"
                                fill="#475569"
                                textAnchor="middle"
                              >
                                {band}
                              </text>
                            )
                          )}

                          <polyline
                            fill="none"
                            stroke="#2563eb"
                            strokeWidth="2.5"
                            strokeLinejoin="round"
                            points={tomatoMatResult.mean_spectrum.map(
                              (point) =>
                                `${65 + (point.band - 1) / 549 * 670},${280 - point.mean * 240}`
                            ).join(" ")}
                          />

                          <text
                            x="400"
                            y="326"
                            fontSize="13"
                            fill="#334155"
                            textAnchor="middle"
                          >
                            Spectral band index
                          </text>
                        </svg>
                      </div>
                    </div>

                    <TomatoSpectralInsights
                      type="matlab"
                      result={tomatoMatResult}
                    />

                    <div className="warning">
                      {tomatoMatResult.warning}
                    </div>
                  </section>
                )}

                {[
                  {
                    src: "/analysis/tomato/tomato_smoothed_normalized_spectral_signatures.png",
                    title: "Normalized Spectral Signatures",
                    insightType: "signatures",
                    description: "Smoothed spectral patterns across the five tomato classes."
                  },
                  {
                    src: "/analysis/tomato/tomato_normalized_spectral_pca.png",
                    title: "Principal Component Analysis",
                    insightType: "pca",
                    description: "PCA visualization of the normalized spectral data."
                  },
                  {
                    src: "/analysis/tomato/tomato_normalized_between_class_variance.png",
                    title: "Between-Class Spectral Variance",
                    insightType: "variance",
                    description: "Bandwise variation among the analyzed class signatures."
                  }
                ].map((plot) => (
                  <div className="prediction-card" key={plot.src}>
                    <h3>{plot.title}</h3>
                    <p>{plot.description}</p>
                    <img
                      src={plot.src}
                      alt={plot.title}
                      style={{
                        display: "block",
                        width: "100%",
                        maxWidth: "100%",
                        height: "auto",
                        marginTop: "16px",
                        borderRadius: "12px"
                      }}
                      loading="lazy"
                    />
                    <TomatoSpectralInsights type={plot.insightType} />
                  </div>
                ))}

                <div className="warning">
                  Exploratory research only. Several fungal-associated
                  classes have highly similar mean spectral signatures.
                  These plots do not establish diagnostic accuracy or
                  food-safety suitability.
                </div>
              </section>
            )}

            {hsiMode === "pistachio" && (
              <>
                <section className="workspace">
                  <div className="section-heading">
                    <div>
                      <div className="eyebrow">MODULE B</div>
                      <h2>Pistachio HSI Analysis</h2>
                      <p>Upload a matching ENVI HDR and BIL file pair.</p>
                    </div>
                    <span className="version">Experimental v0.1</span>
                  </div>

                  <form className="upload-form" onSubmit={analyze}>
                    <div className="upload-grid">
                      <label className="upload-box">
                        <span className="upload-symbol">↑</span>
                        <strong>HDR Metadata</strong>
                        <span>{hdr ? hdr.name : "Select .hdr file"}</span>
                        <input
                          type="file"
                          accept=".hdr"
                          onChange={(e) => {
                            setHdr(e.target.files?.[0] || null);
                            setResult(null);
                          }}
                        />
                      </label>

                      <label className="upload-box">
                        <span className="upload-symbol">↑</span>
                        <strong>Hyperspectral Cube</strong>
                        <span>{bil ? bil.name : "Select .bil file"}</span>
                        <input
                          type="file"
                          accept=".bil"
                          onChange={(e) => {
                            setBil(e.target.files?.[0] || null);
                            setResult(null);
                          }}
                        />
                      </label>
                    </div>

                    <button
                      className="analyze-button"
                      type="submit"
                      disabled={loading || !hdr || !bil}
                    >
                      {loading ? "Analyzing spectral data..." : "Analyze Sample →"}
                    </button>
                    {loading && (
                      <p className="loading-note">
                        Uploading files and running the trained model. Please wait.
                      </p>
                    )}
                    {error && <div className="error-message">{error}</div>}
                  </form>
                </section>

                {prediction && (
                  <section className="results">
                    <div className="section-heading">
                      <div>
                        <div className="eyebrow">ANALYSIS COMPLETE</div>
                        <h2>Prediction Results</h2>
                      </div>
                      <span className="status active">Completed</span>
                    </div>

                    <div className="prediction-card">
                      <span>Exploratory majority-vote classification</span>
                      <h2 className={below ? "below" : "above"}>
                        {below ? "Below Threshold" : "At or Above Threshold"}
                      </h2>
                      <p>AFB1 classification threshold: 8 µg/kg</p>
                    </div>

                    <div className="metrics">
                      <div className="metric">
                        <span>Foreground pixels</span>
                        <strong>{prediction.foreground_pixels?.toLocaleString()}</strong>
                      </div>
                      <div className="metric">
                        <span>Sampled pixels</span>
                        <strong>{prediction.sampled_pixels?.toLocaleString()}</strong>
                      </div>
                      <div className="metric">
                        <span>Pixels predicting at/above</span>
                        <strong>
                          {(prediction.fraction_at_or_above * 100).toFixed(2)}%
                        </strong>
                      </div>
                      <div className="metric">
                        <span>Mean class-1 model probability</span>
                        <strong>
                          {(prediction.mean_model_probability_at_or_above * 100).toFixed(2)}%
                        </strong>
                        <small>Uncalibrated</small>
                      </div>
                    </div>
                    <PistachioSpectralInsights type="prediction" result={result} />
                    <div className="warning">{result.warning}</div>
                  </section>
                )}
              </>
            )}

 
            {hsiMode === "pistachio" && (
              <section className="workspace">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">SPECTRAL RESEARCH</div>
                    <h2>Pistachio Spectral Signatures</h2>
                    <p>
                      Exploratory wavelength analysis associated with
                      AFB1 threshold groups in the study dataset.
                    </p>
                  </div>
                  <span className="version">Research Analysis</span>
                </div>

                <div className="metrics">
                  <div className="metric">
                    <span>Exploratory AFB1 threshold</span>
                    <strong>8 µg/kg</strong>
                  </div>
                  <div className="metric">
                    <span>Prominent analyzed region</span>
                    <strong>774–792 nm</strong>
                  </div>
                  <div className="metric">
                    <span>Analysis type</span>
                    <strong>Spectral comparison</strong>
                  </div>
                </div>

                {[
                  {
                    src: "/analysis/pistachio/pistachio_threshold_spectral_signatures.png",
                    insightType: "signatures",
                    title: "Threshold-Group Spectral Signatures",
                    description: "Comparison of the mean spectral patterns of the study's AFB1 threshold groups."
                  },
                  {
                    src: "/analysis/pistachio/pistachio_threshold_difference.png",
                    insightType: "difference",
                    title: "Threshold-Group Spectral Difference",
                    description: "Wavelength-dependent differences between the two threshold groups."
                  },
                  {
                    src: "/analysis/pistachio/pistachio_afb1_spectral_correlation.png",
                    insightType: "correlation",
                    title: "AFB1-Related Spectral Correlation",
                    description: "Exploratory statistical associations between wavelength features and AFB1 measurements."
                  }
                ].map((plot) => (
                  <div className="prediction-card" key={plot.src}>
                    <h3>{plot.title}</h3>
                    <p>{plot.description}</p>
                    <img
                      src={plot.src}
                      alt={plot.title}
                      style={{
                        display: "block",
                        width: "100%",
                        height: "auto",
                        marginTop: "16px",
                        borderRadius: "12px"
                      }}
                      loading="lazy"
                    />
                    <PistachioSpectralInsights type={plot.insightType} />
                  </div>
                ))}

                <div className="warning">
                  Research visualization only. Spectral correlations
                  do not independently establish AFB1 concentration,
                  contamination status, or food safety.
                </div>
              </section>
            )}

            {["eggplant", "cabbage"].includes(hsiMode) && (
              <>
                <section className="workspace">
                  <div className="section-heading">
                    <div>
                      <div className="eyebrow">MODULE B</div>
                      <h2>{hsiMode === "eggplant" ? "Eggplant" : "Cabbage"} HSI Analysis</h2>
                      <p>Experimental pixel-level N₂ category classification.</p>
                    </div>
                    <span className="version">Experimental</span>
                  </div>

                  <div className="rgb-mode-selector" role="group" aria-label="HSI analysis source">
                    <button type="button"
                      className={`rgb-mode-button ${nitrogenMode === "upload" ? "active" : ""}`}
                      aria-pressed={nitrogenMode === "upload"}
                      disabled={nitrogenLoading || nitrogenUploadLoading}
                      onClick={() => {
                        setNitrogenMode("upload");
                        setNitrogenResult(null);
                        setNitrogenError("");
                      }}>
                      Upload HSI Sample
                    </button>
                    <button type="button"
                      className={`rgb-mode-button ${nitrogenMode === "local" ? "active" : ""}`}
                      aria-pressed={nitrogenMode === "local"}
                      disabled={nitrogenLoading || nitrogenUploadLoading}
                      onClick={() => {
                        setNitrogenMode("local");
                        setNitrogenUploadResult(null);
                        setNitrogenUploadError("");
                      }}>
                      Local Dataset (Developer Test)
                    </button>
                  </div>

                  {nitrogenMode === "upload" ? (
                    <>
                      <p className="loading-note">
                        Upload matching ENVI HDR and spectral binary files containing
                        277 bands and wavelengths compatible with the trained model.
                        Ground-truth labels are not required. HDR limit: 2 MB;
                        binary limit: 100 MB. Large original datasets cannot be uploaded yet.
                      </p>
                      <form className="upload-form" onSubmit={analyzeNitrogenUpload}>
                        <div className="upload-grid">
                          <label className="upload-box">
                            <span className="upload-symbol">↑</span>
                            <strong>HDR Metadata</strong>
                            <span>{nitrogenHdr ? nitrogenHdr.name : "Select .hdr file"}</span>
                            <input key={`${hsiMode}-hdr`} type="file" accept=".hdr"
                              disabled={nitrogenUploadLoading}
                              onChange={(e) => {
                                setNitrogenHdr(e.target.files?.[0] || null);
                                setNitrogenUploadResult(null);
                                setNitrogenUploadError("");
                              }} />
                          </label>
                          <label className="upload-box">
                            <span className="upload-symbol">↑</span>
                            <strong>Hyperspectral Binary</strong>
                            <span>{nitrogenBil ? nitrogenBil.name : "Select .bil or extensionless ENVI data"}</span>
                            <input key={`${hsiMode}-bil`} type="file" accept=".bil,.dat,application/octet-stream"
                              disabled={nitrogenUploadLoading}
                              onChange={(e) => {
                                setNitrogenBil(e.target.files?.[0] || null);
                                setNitrogenUploadResult(null);
                                setNitrogenUploadError("");
                              }} />
                          </label>
                        </div>
                        <button className="analyze-button" type="submit"
                          disabled={nitrogenUploadLoading || !nitrogenHdr || !nitrogenBil}>
                          {nitrogenUploadLoading ? "Uploading and analyzing..." : "Analyze Uploaded Sample →"}
                        </button>
                        {nitrogenUploadLoading && (
                          <p className="loading-note">Uploading the cube and running experimental classification. Please wait.</p>
                        )}
                        {nitrogenUploadError && <div className="error-message">{nitrogenUploadError}</div>}
                      </form>
                    </>
                  ) : (
                    <>
                      <p className="loading-note">
                        Developer demonstration: analyze the registered local reflectance
                        cube using its ground-truth map to select eligible pixels.
                        No upload is required.
                      </p>
                      <button type="button" className="analyze-button"
                        disabled={nitrogenLoading}
                        onClick={() => analyzeNitrogen(hsiMode)}>
                        {nitrogenLoading ? "Analyzing hyperspectral data..." : "Analyze Local Dataset →"}
                      </button>
                      {nitrogenLoading && (
                        <p className="loading-note">Processing the local cube. Please wait.</p>
                      )}
                      {nitrogenError && <div className="error-message">{nitrogenError}</div>}
                    </>
                  )}
                </section>

                {(() => {
                  const data = nitrogenMode === "upload"
                    ? nitrogenUploadResult?.result : nitrogenResult;
                  if (!data) return null;
                  return (
                    <section className="results">
                      <div className="section-heading">
                        <div>
                          <div className="eyebrow">EXPERIMENTAL RESULTS</div>
                          <h2>{data.food} N₂ Categories</h2>
                        </div>
                        <span className="status active">Completed</span>
                      </div>
                      <div className="prediction-card">
                        <span>Random Forest + 30-component PCA</span>
                        <h2>Experimental Category Distribution</h2>
                        <p>
                          {nitrogenMode === "upload"
                            ? "Foreground pixels were selected automatically without ground-truth labels."
                            : "Ground-truth labels were used to select eligible pixels from the local dataset."}
                          {" "}Spatially distributed sampling is
  used across the hyperspectral cube. These predictions
  are experimental and may not represent the entire specimen.
                        </p>
                      </div>
                      <div className="metrics">
                        <div className="metric">
                          <span>Pixels analyzed</span>
                          <strong>{data.pixels_analyzed?.toLocaleString()}</strong>
                        </div>
                        {["Low", "Medium", "High"].map((category) => (
                          <div className="metric" key={category}>
                            <span>{category} category</span>
                            <strong>{data.predicted_percentages?.[category]}%</strong>
                            <small>{data.predicted_counts?.[category]?.toLocaleString()} pixels</small>
                          </div>
                        ))}
                      </div>
                      <NitrogenResultInsights
                        data={data}
                        food={hsiMode}
                        mode={nitrogenMode}
                      />
                      <div className="warning">
                        {data.warning} These percentages may not represent the full
                        specimen and must not be used for food-safety decisions.
                      </div>
                    </section>
                  );
                })()}
              </>
            )}
          </>
        )}

        
        {activeModule === "sensor" && (
          <section className="workspace">
            <div className="section-heading">
              <div>
                <div className="eyebrow">MULTISPECTRAL DATA ANALYSIS</div>
                <h2>Strawberry Sensor Analysis</h2>
                <p>
                  Exploratory analysis of recorded Strawberry sensor
                  measurements using PCA and Isolation Forest.
                </p>
              </div>
              <span className="version">Dataset Analysis</span>
            </div>

            <div className="prediction-card">
              <span>Recorded sensor experiments</span>
              <h2>Isolation Forest Anomaly Analysis</h2>
              <p>
                Anomalies indicate unusual sensor observations,
                not independently verified spoilage or freshness.
              </p>
            </div>

            <div className="metrics">
              <div className="metric">
                <span>Experiment 1 observations</span>
                <strong>127,588</strong>
                <small>95 anomalies ? 0.074%</small>
              </div>
              <div className="metric">
                <span>Experiment 2 observations</span>
                <strong>122,388</strong>
                <small>2,405 anomalies ? 1.965%</small>
              </div>
              <div className="metric">
                <span>Analysis methods</span>
                <strong>PCA + Isolation Forest</strong>
              </div>
            </div>
              <StrawberrySensorInsights type="experiments" />



            <div className="prediction-card">
              <h3>Upload Strawberry Sensor CSV</h3>
              <p>
                Analyze recorded Strawberry sensor measurements using
                the trained StandardScaler, PCA, and Isolation Forest.
                The CSV must contain the 63 original sensor features.
              </p>
            </div>

            <form className="upload-form" onSubmit={analyzeSensor}>
              <div className="upload-grid">
                <label className="upload-box">
                  <span className="upload-symbol">?</span>
                  <strong>Strawberry Sensor CSV</strong>
                  <span>
                    {sensorFile ? sensorFile.name : "Select a .csv file"}
                  </span>
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    disabled={sensorLoading}
                    onChange={(event) => {
                      setSensorFile(event.target.files?.[0] || null);
                      setSensorResult(null);
                      setSensorError("");
                    }}
                  />
                </label>
              </div>

              <p className="loading-note">
                Maximum CSV size: 25 MB. Maximum 100,000 observations.
                Missing or non-numeric sensor measurements are rejected.
              </p>

              <button
                className="analyze-button"
                type="submit"
                disabled={!sensorFile || sensorLoading}
              >
                {sensorLoading
                  ? "Analyzing Strawberry sensor data..."
                  : "Analyze Sensor CSV ?"}
              </button>

              {sensorLoading && (
                <p className="loading-note">
                  Applying PCA and Isolation Forest anomaly detection.
                  Please wait.
                </p>
              )}

              {sensorError && (
                <div className="error-message">{sensorError}</div>
              )}
            </form>

            {sensorResult && (
              <section className="results">
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">UPLOAD ANALYSIS COMPLETE</div>
                    <h2>Strawberry Sensor Results</h2>
                    <p>{sensorResult.filename}</p>
                  </div>
                  <span className="status active">Completed</span>
                </div>

                <div className="prediction-card">
                  <span>Unsupervised anomaly detection</span>
                  <h2>
                    {sensorResult.anomaly_percentage.toFixed(2)}% Anomalies
                  </h2>
                  <p>
                    {sensorResult.anomaly_observations.toLocaleString()}
                    {" "}unusual observations detected in the uploaded CSV.
                  </p>
                </div>

                <div className="metrics">
                  <div className="metric">
                    <span>Total observations</span>
                    <strong>
                      {sensorResult.total_observations.toLocaleString()}
                    </strong>
                  </div>

                  <div className="metric">
                    <span>Normal observations</span>
                    <strong>
                      {sensorResult.normal_observations.toLocaleString()}
                    </strong>
                  </div>

                  <div className="metric">
                    <span>Anomalies detected</span>
                    <strong>
                      {sensorResult.anomaly_observations.toLocaleString()}
                    </strong>
                  </div>

                  <div className="metric">
                    <span>Mean anomaly score</span>
                    <strong>
                      {sensorResult.mean_anomaly_score.toFixed(4)}
                    </strong>
                  </div>
                </div>
                <StrawberrySensorInsights type="results" result={sensorResult} />


                <div className="warning">
                  {sensorResult.interpretation}
                </div>
              </section>
            )}

            {[
              {
                src: "/analysis/strawberry/strawberry_pc1_pc2_trajectory.png",
                insightType: "pca",
                title: "PCA Sensor Trajectory",
                description: "Observations projected onto the first two principal components."
              },
              {
                src: "/analysis/strawberry/strawberry_sensor_condition_trend.png",
                insightType: "condition",
                title: "Sensor Condition Trend",
                description: "Exploratory trends in recorded sensor conditions."
              },
              {
                src: "/analysis/strawberry/strawberry_pc1_temporal_trend.png",
                insightType: "temporal",
                title: "Temporal PCA Trend",
                description: "Changes in the first principal-component scores over observations."
              }
            ].map((plot) => (
              <div className="prediction-card" key={plot.src}>
                <h3>{plot.title}</h3>
                <p>{plot.description}</p>
                <img
                  src={plot.src}
                  alt={plot.title}
                  style={{
                    display: "block",
                    width: "100%",
                    height: "auto",
                    marginTop: "16px",
                    borderRadius: "12px"
                  }}
                  loading="lazy"
                />
                <StrawberrySensorInsights type={plot.insightType} />
              </div>
            ))}

            <div className="warning">
              Research dataset analysis only. No physical sensor
              integration or live sensor inference is implemented.
              Anomaly flags are not validated spoilage indicators.
            </div>
          </section>
        )}

        <footer>
          SpectraGuard · AI Food Quality Platform · Research Prototype
        </footer>
      </main>
    </div>
  );
}
