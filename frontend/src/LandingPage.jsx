import { useEffect, useRef, useState } from "react";
import "./LandingPage.css";

const features = [
  {
    number: "01",
    icon: "camera",
    title: "RGB Computer Vision",
    description: "Explore visual freshness and mold-related patterns using image-based AI research.",
    tags: ["Image Analysis", "Deep Learning"],
  },
  {
    number: "02",
    icon: "spectrum",
    title: "Hyperspectral Intelligence",
    description: "Investigate spectral signatures and experimental machine-learning predictions across food datasets.",
    tags: ["Spectral Analysis", "PCA"],
  },
  {
    number: "03",
    icon: "activity",
    title: "Sensor Analytics",
    description: "Explore strawberry sensor observations using dimensionality reduction and anomaly detection.",
    tags: ["Isolation Forest", "Research"],
  },
];

const spectrum = Array.from({ length: 54 }, (_, i) => ({
  height: 18 + Math.abs(Math.sin(i * 0.31) * Math.cos(i * 0.13)) * 75,
  delay: i * 0.028,
}));


function LandingIcon({ name, size = 22 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
  };

  const paths = {
    sparkle: (
      <>
        <path d="M12 2l2.4 7.6L22 12l-7.6 2.4L12 22l-2.4-7.6L2 12l7.6-2.4L12 2z" />
      </>
    ),
    arrow: (
      <>
        <path d="M5 19L19 5" />
        <path d="M8 5h11v11" />
      </>
    ),
    down: (
      <>
        <path d="M12 4v16" />
        <path d="M5 13l7 7 7-7" />
      </>
    ),
    camera: (
      <>
        <rect x="3" y="6" width="18" height="14" rx="3" />
        <circle cx="12" cy="13" r="4" />
        <path d="M8 6l1.5-2h5L16 6" />
      </>
    ),
    spectrum: (
      <>
        <path d="M3 18V9M7 18V5M11 18V11M15 18V3M19 18V8" />
        <path d="M2 21h20" />
      </>
    ),
    activity: (
      <>
        <path d="M2 12h4l3-7 5 14 3-7h5" />
      </>
    ),
  };

  return <svg {...common}>{paths[name] || paths.sparkle}</svg>;
}

export default function LandingPage({ onGetStarted }) {
  const [scrolled, setScrolled] = useState(false);
  const [activeFeature, setActiveFeature] = useState(1);
  const [menuOpen, setMenuOpen] = useState(false);
  const heroRef = useRef(null);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 30);
    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("sg-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 }
    );

    document.querySelectorAll(".sg-reveal").forEach((el) => observer.observe(el));

    return () => {
      window.removeEventListener("scroll", handleScroll);
      observer.disconnect();
    };
  }, []);

  const handlePointer = (event) => {
    if (!heroRef.current) return;
    const rect = heroRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    heroRef.current.style.setProperty("--pointer-x", `${x}%`);
    heroRef.current.style.setProperty("--pointer-y", `${y}%`);
  };

  const navigate = (id) => {
    setMenuOpen(false);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="sg-landing">
      <nav className={`sg-nav ${scrolled ? "sg-nav-scrolled" : ""}`}>
        <a className="sg-logo" href="#home" aria-label="SpectraGuard home">
          <span className="sg-logo-symbol"><LandingIcon name="sparkle" size={21} /></span>
          <span>Spectra<span className="sg-logo-accent">Guard</span></span>
        </a>

        <button
          className="sg-menu-toggle"
          type="button"
          aria-label="Toggle navigation"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? "Close" : "Menu"}
        </button>

        <div className={`sg-nav-links ${menuOpen ? "sg-nav-open" : ""}`}>
          <button type="button" onClick={() => navigate("technology")}>Technology</button>
          <button type="button" onClick={() => navigate("workflow")}>How It Works</button>
          <button type="button" onClick={() => navigate("about")}>About</button>
          <button className="sg-nav-login" type="button" onClick={onGetStarted}>
            Login <span><LandingIcon name="arrow" size={17} /></span>
          </button>
        </div>
      </nav>

      <main>
        <section
          className="sg-hero"
          id="home"
          ref={heroRef}
          onPointerMove={handlePointer}
        >
          <div className="sg-hero-glow" />
          <div className="sg-grid-overlay" />
          <div className="sg-orbit sg-orbit-one" />
          <div className="sg-orbit sg-orbit-two" />

          <div className="sg-hero-content">
            <div className="sg-eyebrow">
              <span className="sg-live-dot" />
              MULTIMODAL FOOD QUALITY RESEARCH
            </div>

            <h1>
              See Beyond
              <br />
              <span className="sg-gradient-text">the Surface.</span>
            </h1>

            <p className="sg-hero-description">
              Discover the hidden patterns in food through computer vision,
              hyperspectral imaging, and intelligent sensor analytics.
              One research platform. Multiple perspectives.
            </p>

            <div className="sg-hero-actions">
              <button className="sg-primary-button" type="button" onClick={onGetStarted}>
                Get Started <span><LandingIcon name="arrow" size={17} /></span>
              </button>
              <button
                className="sg-secondary-button"
                type="button"
                onClick={() => navigate("technology")}
              >
                Explore Technology <span><LandingIcon name="down" size={17} /></span>
              </button>
            </div>

            <div className="sg-hero-footnote">
              <span className="sg-footnote-line" />
              RESEARCH PROTOTYPE | AI-POWERED ANALYSIS
            </div>
          </div>

          <div className="sg-visual" aria-label="Animated spectral scanning visualization">
            <div className="sg-visual-frame">
              <div className="sg-visual-topline">
                <span><span className="sg-live-dot" /> SPECTRAL EXPLORER</span>
                <span>LIVE VISUALIZATION</span>
              </div>

              <div className="sg-scan-stage">
                <div className="sg-fruit sg-fruit-back sg-apple" aria-label="Apple" />
                <div className="sg-fruit sg-fruit-main sg-strawberry" aria-label="Strawberry" />
                <div className="sg-fruit sg-fruit-front sg-tomato" aria-label="Tomato" />
                <div className="sg-scan-line" />
                <div className="sg-scan-crosshair sg-crosshair-one" />
                <div className="sg-scan-crosshair sg-crosshair-two" />
                <div className="sg-scan-label">RGB + SPECTRAL</div>
              </div>

              <div className="sg-spectrum-panel">
                <div className="sg-spectrum-heading">
                  <span>SPECTRAL RESPONSE</span>
                  <span>EXPLORATORY SIGNAL</span>
                </div>
                <div className="sg-spectrum-bars">
                  {spectrum.map((bar, index) => (
                    <span
                      key={index}
                      style={{
                        "--bar-height": `${bar.height}%`,
                        "--bar-delay": `${bar.delay}s`,
                      }}
                    />
                  ))}
                </div>
                <div className="sg-spectrum-axis">
                  <span>VISIBLE</span>
                  <span>SPECTRAL FEATURES</span>
                </div>
              </div>
            </div>

            <div className="sg-floating-tag sg-tag-one">
              <span className="sg-tag-icon"><LandingIcon name="sparkle" size={15} /></span> Computer Vision
            </div>
            <div className="sg-floating-tag sg-tag-two">
              <span className="sg-tag-icon"><LandingIcon name="sparkle" size={15} /></span> Spectral Insights
            </div>
          </div>

          <button
            className="sg-scroll-cue"
            type="button"
            onClick={() => navigate("technology")}
          >
            <span>SCROLL TO EXPLORE</span>
            <span className="sg-scroll-arrow"><LandingIcon name="down" size={18} /></span>
          </button>
        </section>

        <section className="sg-technology" id="technology">
          <div className="sg-section-intro sg-reveal">
            <span className="sg-section-kicker">THE TECHNOLOGY</span>
            <h2>Three ways to <span>look deeper.</span></h2>
            <p>
              Different data modalities reveal different patterns.
              Explore how SpectraGuard brings them together in one platform.
            </p>
          </div>

          <div className="sg-feature-grid">
            {features.map((feature, index) => (
              <button
                type="button"
                className={`sg-feature-card sg-reveal ${
                  activeFeature === index ? "sg-feature-active" : ""
                }`}
                key={feature.number}
                onMouseEnter={() => setActiveFeature(index)}
                onFocus={() => setActiveFeature(index)}
                onClick={() => setActiveFeature(index)}
              >
                <div className="sg-feature-top">
                  <span>{feature.number} / 03</span>
                  <span className="sg-feature-arrow"><LandingIcon name="arrow" /></span>
                </div>
                <div className="sg-feature-icon"><LandingIcon name={feature.icon} size={43} /></div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
                <div className="sg-feature-tags">
                  {feature.tags.map((tag) => <span key={tag}>{tag}</span>)}
                </div>
                <div className="sg-feature-line" />
              </button>
            ))}
          </div>
        </section>

        <section className="sg-workflow" id="workflow">
          <div className="sg-workflow-inner">
            <div className="sg-workflow-copy sg-reveal">
              <span className="sg-section-kicker">THE WORKFLOW</span>
              <h2>From raw data<br />to <span>deeper insight.</span></h2>
              <p>
                A streamlined research workflow designed to make
                experimental food analysis more accessible.
              </p>
              <button className="sg-text-button" type="button" onClick={onGetStarted}>
                Enter the platform <span><LandingIcon name="arrow" size={17} /></span>
              </button>
            </div>

            <div className="sg-steps">
              {[
                ["01", "Choose a modality", "Select RGB, hyperspectral, or sensor analysis."],
                ["02", "Provide your data", "Upload a compatible image, spectral cube, or sensor CSV."],
                ["03", "Explore the results", "Review model outputs, visualizations, and research limitations."],
              ].map(([number, title, description]) => (
                <div className="sg-step sg-reveal" key={number}>
                  <span className="sg-step-number">{number}</span>
                  <div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                  </div>
                  <span className="sg-step-symbol"><LandingIcon name="arrow" /></span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="sg-final" id="about">
          <div className="sg-final-glow" />
          <div className="sg-final-content sg-reveal">
            <span className="sg-section-kicker">EXPLORE SPECTRAGUARD</span>
            <h2>Curiosity meets<br /><span>intelligence.</span></h2>
            <p>
              Step into a multimodal AI research environment and
              explore what food data can reveal.
            </p>
            <button className="sg-primary-button" type="button" onClick={onGetStarted}>
              Launch SpectraGuard <span><LandingIcon name="arrow" size={17} /></span>
            </button>
            <p className="sg-disclaimer">
              Experimental research platform. Predictions are not
              certified food-safety determinations.
            </p>
          </div>
        </section>
      </main>

      <footer className="sg-footer">
        <div className="sg-logo">
          <span className="sg-logo-symbol"><LandingIcon name="sparkle" size={21} /></span>
          <span>Spectra<span className="sg-logo-accent">Guard</span></span>
        </div>
        <span>AI Food Quality Platform | Research Prototype</span>
        <span>? {new Date().getFullYear()} SpectraGuard</span>
      </footer>
    </div>
  );
}
