"""TFRecord episodes may omit optional camera fields per step."""

from tfrecord.writer import TFRecordWriter

from crane_x7_vla.core.data.tfrecord_reader import TFRecordReader


def test_optional_cameras_do_not_drop_or_duplicate_steps(tmp_path):
    path = tmp_path / "episode.tfrecord"
    writer = TFRecordWriter(str(path))
    try:
        for step in range(2):
            fields = {
                "observation/proprio": ([0.0] * 8, "float"),
                "observation/image_primary": (b"image", "byte"),
                "action": ([float(step)] * 8, "float"),
                "task/language_instruction": (b"move", "byte"),
                "dataset_name": (b"crane_x7", "byte"),
            }
            if step == 0:
                fields["observation/image_wrist"] = (b"wrist", "byte")
            writer.write(fields)
    finally:
        writer.close()

    reader = TFRecordReader(
        [path],
        feature_spec={"observation/image_primary": "byte", "observation/image_wrist": "byte"},
        use_alternative_keys=True,
    )
    examples = list(reader)

    assert reader.count_records() == 2
    assert len(examples) == 2
    assert "observation/image_wrist" in examples[0]
    assert "observation/image_wrist" not in examples[1]
    assert examples[0]["action"][0] == 0
    assert examples[1]["action"][0] == 1
