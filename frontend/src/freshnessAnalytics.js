const SUPPORTED_FOODS = [
  "Apple",
  "Banana",
  "Strawberry",
  "Tomato",
];

function normalizeDate(value) {
  if (!value) return null;

  const normalized = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value)
    ? value
    : `${value}Z`;

  const date = new Date(normalized);

  return Number.isNaN(date.getTime()) ? null : date;
}

function getFreshnessRecord(record) {
  if (
    String(record?.module || "").toLowerCase() !== "rgb" ||
    String(record?.analysis_type || "").toLowerCase() !== "freshness"
  ) {
    return null;
  }

  const result = record.result_data?.result || {};

  const prediction = String(
    result.prediction || record.prediction || ""
  );

  let food = String(
    result.food || record.food_name || ""
  );

  let freshness = String(result.freshness || "");

  if (!freshness && prediction.includes("_")) {
    const parts = prediction.split("_");
    food = parts[0];
    freshness = parts[1];
  }

  food = SUPPORTED_FOODS.find(
    (item) => item.toLowerCase() === food.toLowerCase()
  );

  freshness = ["Fresh", "Rotten"].find(
    (item) => item.toLowerCase() === freshness.toLowerCase()
  );

  if (!food || !freshness) return null;

  return {
    id: record.id,
    food,
    freshness,
    createdAt: normalizeDate(record.created_at),
    original: record,
  };
}

export function calculateFreshnessAnalytics(
  records,
  period = "all",
  now = new Date()
) {
  const referenceDate = new Date(now);

  if (Number.isNaN(referenceDate.getTime())) {
    throw new Error("Invalid reference date.");
  }

  const cutoff = new Date(referenceDate);

  if (period === "7d") {
    cutoff.setDate(cutoff.getDate() - 7);
  } else if (period === "30d") {
    cutoff.setDate(cutoff.getDate() - 30);
  } else if (period !== "all") {
    throw new Error("Unsupported analytics period.");
  }

  const normalized = records
    .map(getFreshnessRecord)
    .filter(Boolean);

  const filtered = normalized.filter((record) => {
    if (period === "all") return true;

    return (
      record.createdAt !== null &&
      record.createdAt >= cutoff &&
      record.createdAt <= referenceDate
    );
  });

  const byFruit = SUPPORTED_FOODS.map((food) => {
    const matching = filtered.filter(
      (record) => record.food === food
    );

    return {
      food,
      fresh: matching.filter(
        (record) => record.freshness === "Fresh"
      ).length,
      rotten: matching.filter(
        (record) => record.freshness === "Rotten"
      ).length,
      total: matching.length,
    };
  });

  const dailyCounts = new Map();

  for (const record of filtered) {
    if (!record.createdAt) continue;

    const day = record.createdAt
      .toISOString()
      .slice(0, 10);

    dailyCounts.set(
      day,
      (dailyCounts.get(day) || 0) + 1
    );
  }

  const timeline = Array.from(
    dailyCounts,
    ([date, count]) => ({ date, count })
  ).sort((a, b) => a.date.localeCompare(b.date));

  const recent = [...filtered]
    .sort((a, b) => {
      const first = a.createdAt?.getTime() ?? 0;
      const second = b.createdAt?.getTime() ?? 0;

      return second - first || b.id - a.id;
    })
    .slice(0, 10);

  return {
    period,
    total: filtered.length,
    fresh: filtered.filter(
      (record) => record.freshness === "Fresh"
    ).length,
    rotten: filtered.filter(
      (record) => record.freshness === "Rotten"
    ).length,
    byFruit,
    timeline,
    recent,
    excludedRecords: records.length - normalized.length,
  };
}
