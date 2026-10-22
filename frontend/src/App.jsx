import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const HSI_FOODS = ["pistachio", "eggplant", "cabbage"];

export default function App() {
  const [activeModule, setActiveModule] = useState("hsi");
  const [rgbMode, setRgbMode] = useState("freshness");
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
      const response = await fetch(`${API_URL}/predict/hsi/pistachio`, {
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
      const response = await fetch(`${API_URL}/predict/hsi/${food}/local`, {
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
      const response = await fetch(`${API_URL}/predict/hsi/${hsiMode}/upload`, {
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
      const response = await fetch(`${API_URL}${endpoint}`, {
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

          <article className="module-card">
            <div className="module-icon">⌁</div>
            <h3>Multispectral Sensor</h3>
            <p>Analyze AS7341 spectral sensor measurements.</p>
            <span className="status muted">Hardware integration pending</span>
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
                    <div className="warning">{result.warning}</div>
                  </section>
                )}
              </>
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

        <footer>
          SpectraGuard · AI Food Quality Platform · Research Prototype
        </footer>
      </main>
    </div>
  );
}
