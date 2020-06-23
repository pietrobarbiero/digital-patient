import unittest


class TestTemplateObject(unittest.TestCase):
    def test_object(self):
        import digital_patient

        t = digital_patient.TemplateObject()
        self.assertTrue(isinstance(t, digital_patient.TemplateObject))
        return


if __name__ == '__main__':
    unittest.main()
