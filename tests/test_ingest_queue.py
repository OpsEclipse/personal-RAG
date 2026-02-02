import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services import ingest_queue
from app.services.file_storage import UPLOADS_DIR, cleanup_job_files, save_uploaded_file


class TestIngestQueue(unittest.TestCase):
    def setUp(self) -> None:
        ingest_queue.JOB_STORE.clear()
        ingest_queue.JOB_QUEUE.clear()

    def test_validate_ingest_request_rejects_empty_values(self) -> None:
        with self.assertRaises(ValueError):
            ingest_queue.validate_ingest_request("", "namespace", "index", ingest_queue.RoutingMode.MANUAL)

        with self.assertRaises(ValueError):
            ingest_queue.validate_ingest_request("file.exe", "namespace", "index", ingest_queue.RoutingMode.MANUAL)

        with self.assertRaises(ValueError):
            ingest_queue.validate_ingest_request("file.pdf", "", "index", ingest_queue.RoutingMode.MANUAL)

        with self.assertRaises(ValueError):
            ingest_queue.validate_ingest_request("file.pdf", "namespace", " ", ingest_queue.RoutingMode.MANUAL)

        with self.assertRaises(ValueError):
            ingest_queue.validate_ingest_request(
                "file.pdf", "namespace", "index", ingest_queue.RoutingMode.AUTO
            )

        with self.assertRaises(ValueError):
            ingest_queue.validate_ingest_request(
                "file.pdf", "namespace", "index", ingest_queue.RoutingMode.PER_CHUNK
            )

    def test_parse_metadata_json(self) -> None:
        self.assertIsNone(ingest_queue.parse_metadata_json(None))
        self.assertIsNone(ingest_queue.parse_metadata_json("  "))
        self.assertEqual(
            ingest_queue.parse_metadata_json('{"source": "unit-test"}'),
            {"source": "unit-test"},
        )

        with self.assertRaises(ValueError):
            ingest_queue.parse_metadata_json("{bad json}")

        with self.assertRaises(ValueError):
            ingest_queue.parse_metadata_json('["not-an-object"]')

    def test_add_file_to_queue_persists_record(self) -> None:
        job_id = str(uuid.uuid4())
        ingest_queue.add_file_to_queue(
            job_id=job_id,
            filename="file.pdf",
            content_type="application/pdf",
            file_path="/tmp/test/file.pdf",
            namespace="namespace",
            index="index",
            routing_mode=ingest_queue.RoutingMode.MANUAL,
            metadata={"source": "unit-test"},
        )
        record = ingest_queue.JOB_STORE.get(job_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.filename, "file.pdf")
        self.assertEqual(record.file_path, "/tmp/test/file.pdf")
        self.assertEqual(record.metadata["namespace"], "namespace")
        self.assertEqual(record.metadata["index"], "index")
        self.assertEqual(record.metadata["routing_mode"], "manual")
        self.assertEqual(record.metadata["metadata"], {"source": "unit-test"})
        self.assertIn(job_id, ingest_queue.JOB_QUEUE)

    def test_validate_job_id_rejects_invalid_id(self) -> None:
        with self.assertRaises(ValueError):
            ingest_queue.validate_job_id("not-a-uuid")


class TestFileStorage(unittest.TestCase):
    def setUp(self) -> None:
        self.test_job_id = str(uuid.uuid4())
        self.test_job_dir = UPLOADS_DIR / self.test_job_id

    def tearDown(self) -> None:
        if self.test_job_dir.exists():
            shutil.rmtree(self.test_job_dir)

    def test_save_uploaded_file(self) -> None:
        mock_file = MagicMock()
        mock_file.filename = "test.pdf"
        mock_file.file.read.return_value = b"test content"

        result_path = save_uploaded_file(self.test_job_id, mock_file)

        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(Path(result_path).name, "test.pdf")
        with open(result_path, "rb") as f:
            self.assertEqual(f.read(), b"test content")
        mock_file.file.seek.assert_called_once_with(0)

    def test_cleanup_job_files(self) -> None:
        self.test_job_dir.mkdir(parents=True, exist_ok=True)
        test_file = self.test_job_dir / "test.pdf"
        test_file.write_bytes(b"test content")

        self.assertTrue(self.test_job_dir.exists())
        cleanup_job_files(self.test_job_id)
        self.assertFalse(self.test_job_dir.exists())

    def test_cleanup_job_files_nonexistent(self) -> None:
        cleanup_job_files("nonexistent-job-id")


class TestProcessJob(unittest.TestCase):
    def setUp(self) -> None:
        ingest_queue.JOB_STORE.clear()
        ingest_queue.JOB_QUEUE.clear()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("Test content for processing.")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_job_nonexistent(self) -> None:
        ingest_queue.process_job("nonexistent-job-id")

    def test_process_job_no_file_path(self) -> None:
        job_id = str(uuid.uuid4())
        ingest_queue.JOB_STORE[job_id] = ingest_queue.IngestJobRecord(
            job_id=job_id,
            filename="test.pdf",
            status="queued",
            metadata={"namespace": "test", "index": "test-index"},
        )

        ingest_queue.process_job(job_id)

        record = ingest_queue.JOB_STORE[job_id]
        self.assertEqual(record.status, "failed")
        self.assertIn("No file path", record.error)

    @patch("app.services.ingest_queue.upsert_vectors")
    @patch("app.services.ingest_queue.embed_texts_batched")
    @patch("app.services.ingest_queue.cleanup_job_files")
    def test_process_job_success(
        self, mock_cleanup: MagicMock, mock_embed: MagicMock, mock_upsert: MagicMock
    ) -> None:
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_upsert.return_value = 1

        job_id = str(uuid.uuid4())
        ingest_queue.JOB_STORE[job_id] = ingest_queue.IngestJobRecord(
            job_id=job_id,
            filename="test.txt",
            file_path=self.test_file,
            status="queued",
            metadata={
                "namespace": "test-ns",
                "index": "test-index",
                "metadata": {"source_url": "https://example.com"},
            },
        )
        ingest_queue.JOB_QUEUE.append(job_id)

        ingest_queue.process_job(job_id)

        record = ingest_queue.JOB_STORE[job_id]
        self.assertEqual(record.status, "completed")
        self.assertIsNone(record.error)
        self.assertIsNotNone(record.chunks_processed)
        self.assertGreater(record.chunks_processed, 0)
        mock_cleanup.assert_called_once_with(job_id)
        mock_upsert.assert_called_once()
        call_args = mock_upsert.call_args
        self.assertEqual(call_args[0][0], "test-index")
        self.assertEqual(call_args[0][1], "test-ns")

    @patch("app.services.ingest_queue.embed_texts_batched")
    def test_process_job_embedding_failure(self, mock_embed: MagicMock) -> None:
        mock_embed.side_effect = Exception("Embedding API error")

        job_id = str(uuid.uuid4())
        ingest_queue.JOB_STORE[job_id] = ingest_queue.IngestJobRecord(
            job_id=job_id,
            filename="test.txt",
            file_path=self.test_file,
            status="queued",
            metadata={"namespace": "test", "index": "test-index"},
        )

        ingest_queue.process_job(job_id)

        record = ingest_queue.JOB_STORE[job_id]
        self.assertEqual(record.status, "failed")
        self.assertIn("Embedding API error", record.error)

    @patch("app.services.ingest_queue.embed_texts_batched")
    def test_process_job_no_index(self, mock_embed: MagicMock) -> None:
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        job_id = str(uuid.uuid4())
        ingest_queue.JOB_STORE[job_id] = ingest_queue.IngestJobRecord(
            job_id=job_id,
            filename="test.txt",
            file_path=self.test_file,
            status="queued",
            metadata={"namespace": "test"},
        )

        ingest_queue.process_job(job_id)

        record = ingest_queue.JOB_STORE[job_id]
        self.assertEqual(record.status, "failed")
        self.assertIn("No index name", record.error)


if __name__ == "__main__":
    unittest.main()
