(function () {
  var form = document.getElementById("uploadForm");
  var overlay = document.getElementById("loadingOverlay");
  var progressFill = document.getElementById("progressFill");
  var loadingError = document.getElementById("loadingError");
  var dismissBtn = document.getElementById("dismissBtn");
  var submitBtn = document.getElementById("submitBtn");

  var progressInterval = null;
  var targetPercent = 0;

  function setProgress(percent) {
    targetPercent = Math.min(100, Math.max(0, percent));
    progressFill.style.width = targetPercent + "%";
    progressFill.setAttribute("aria-valuenow", Math.round(targetPercent));
  }

  function hideOverlay() {
    overlay.classList.remove("active");
    loadingError.style.display = "none";
    dismissBtn.style.display = "none";
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
    setProgress(0);
  }

  function showError(msg) {
    loadingError.textContent = msg;
    loadingError.style.display = "block";
    dismissBtn.style.display = "inline-block";
    setProgress(0);
  }

  function runSimulatedProgress() {
    var start = 0;
    var duration = 8000;
    var startTime = Date.now();
    progressInterval = setInterval(function () {
      var elapsed = Date.now() - startTime;
      var p = Math.min(90, (elapsed / duration) * 90);
      setProgress(p);
    }, 150);
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var fileInput = document.getElementById("file");
    if (!fileInput.files || !fileInput.files.length) return;

    overlay.classList.add("active");
    loadingError.style.display = "none";
    dismissBtn.style.display = "none";
    setProgress(0);
    runSimulatedProgress();

    var formData = new FormData(form);
    var xhr = new XMLHttpRequest();

    xhr.open("POST", form.action);
    xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
    xhr.responseType = "json";

    xhr.onload = function () {
      if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
      }
      setProgress(100);

      var res = xhr.response;
      if (!res || typeof res !== "object") {
        showError("Invalid response from server.");
        return;
      }
      if (res.success && res.redirect) {
        setTimeout(function () {
          window.location.href = res.redirect;
        }, 300);
        return;
      }
      showError(res.message || "Something went wrong.");
    };

    xhr.onerror = function () {
      if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
      }
      showError("Network error. Please try again.");
    };

    xhr.send(formData);
  });

  dismissBtn.addEventListener("click", hideOverlay);
})();
