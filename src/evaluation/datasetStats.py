from dataclasses import asdict, dataclass, field


@dataclass
class DatasetStats:
    loaded_per_level: dict[str, int] = field(default_factory=dict)
    kept_per_level: dict[str, int] = field(default_factory=dict)

    skipped_similar: int = 0
    skipped_length_ratio: int = 0
    skipped_duplicate: int = 0

    similarity_scores: list[float] = field(default_factory=list)
    length_ratios: list[float] = field(default_factory=list)

    def add_loaded(self, level: str, count: int):
        self.loaded_per_level[level] = self.loaded_per_level.get(level, 0) + count

    def add_kept(self, level: str):
        self.kept_per_level[level] = self.kept_per_level.get(level, 0) + 1

    def to_dict(self) -> dict:
        data = asdict(self)

        data["total_loaded"] = sum(self.loaded_per_level.values())
        data["total_kept"] = sum(self.kept_per_level.values())

        if self.similarity_scores:
            similarity_avg = sum(self.similarity_scores) / len(self.similarity_scores)
            data["similarity_avg"] = similarity_avg
            data["copy_rate"] = similarity_avg
            data["edit_rate"] = 1.0 - similarity_avg
            data["similarity_min"] = min(self.similarity_scores)
            data["similarity_max"] = max(self.similarity_scores)

        if self.length_ratios:
            data["length_ratio_avg"] = sum(self.length_ratios) / len(self.length_ratios)
            data["length_ratio_min"] = min(self.length_ratios)
            data["length_ratio_max"] = max(self.length_ratios)

        return data
