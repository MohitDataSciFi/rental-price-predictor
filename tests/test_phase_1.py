import asyncio
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

from src.data_loader import DataLoader


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'price': [1000, 1500, 2000, 2500, 3000, 3500],
        'sqft': [500, 750, 1000, 1250, 1500, 1750],
        'bedrooms': [1, 2, 2, 3, 3, 4],
        'bathrooms': [1, 1, 2, 2, 2, 3]
    })


@pytest.fixture
def data_loader(tmp_path):
    """Create a DataLoader instance with a temporary data file."""
    data_file = tmp_path / "test_data.csv"
    data_file.write_text("price,sqft,bedrooms,bathrooms\n1000,500,1,1\n1500,750,2,1\n2000,1000,2,2\n2500,1250,3,2\n3000,1500,3,2\n3500,1750,4,3\n")
    return DataLoader(str(data_file), test_size=0.33, random_state=42)


class TestDataLoaderInitialization:
    """Test DataLoader initialization and validation."""

    def test_valid_initialization(self, tmp_path):
        """Test that valid parameters create a DataLoader instance."""
        data_file = tmp_path / "test.csv"
        data_file.write_text("test")
        loader = DataLoader(str(data_file), test_size=0.2, random_state=42)
        assert loader.data_path == Path(data_file)
        assert loader.test_size == 0.2
        assert loader.random_state == 42

    def test_missing_file_raises_error(self, tmp_path):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DataLoader(str(tmp_path / "nonexistent.csv"))

    def test_invalid_test_size(self, tmp_path):
        """Test that invalid test_size raises ValueError."""
        data_file = tmp_path / "test.csv"
        data_file.write_text("test")
        with pytest.raises(ValueError):
            DataLoader(str(data_file), test_size=0)
        with pytest.raises(ValueError):
            DataLoader(str(data_file), test_size=1)

    def test_invalid_random_state(self, tmp_path):
        """Test that negative random_state raises ValueError."""
        data_file = tmp_path / "test.csv"
        data_file.write_text("test")
        with pytest.raises(ValueError):
            DataLoader(str(data_file), random_state=-1)


class TestDataLoaderLoadData:
    """Test the load_data method."""

    @pytest.mark.asyncio
    async def test_load_data_success(self, data_loader):
        """Test successful data loading."""
        df = await data_loader.load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert list(df.columns) == ['price', 'sqft', 'bedrooms', 'bathrooms']

    @pytest.mark.asyncio
    async def test_load_data_with_mock(self, data_loader):
        """Test data loading with mocked file reading."""
        mock_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        with patch('pandas.read_csv', return_value=mock_df) as mock_read:
            df = await data_loader.load_data()
            mock_read.assert_called_once_with(data_loader.data_path)
            assert df.equals(mock_df)

    @pytest.mark.asyncio
    async def test_load_data_error_handling(self, data_loader):
        """Test error handling when file reading fails."""
        with patch('pandas.read_csv', side_effect=Exception("File corrupted")):
            with pytest.raises(Exception, match="File corrupted"):
                await data_loader.load_data()


class TestDataLoaderSplitData:
    """Test the split_data method."""

    @pytest.mark.asyncio
    async def test_split_data_success(self, data_loader, sample_dataframe):
        """Test successful data splitting."""
        X_train, X_test, y_train, y_test = await data_loader.split_data(
            sample_dataframe, target_col='price'
        )
        
        # Check shapes
        assert len(X_train) == 4
        assert len(X_test) == 2
        assert len(y_train) == 4
        assert len(y_test) == 2
        
        # Check that target column is removed from features
        assert 'price' not in X_train.columns
        assert 'price' not in X_test.columns
        
        # Check that features are preserved
        assert list(X_train.columns) == ['sqft', 'bedrooms', 'bathrooms']
        
        # Check that target values are correct
        assert set(y_train).issubset(set(sample_dataframe['price']))
        assert set(y_test).issubset(set(sample_dataframe['price']))

    @pytest.mark.asyncio
    async def test_split_data_missing_target(self, data_loader, sample_dataframe):
        """Test error when target column doesn't exist."""
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            await data_loader.split_data(sample_dataframe, target_col='nonexistent')

    @pytest.mark.asyncio
    async def test_split_data_reproducibility(self, data_loader, sample_dataframe):
        """Test that splitting is reproducible with same random_state."""
        X_train1, X_test1, y_train1, y_test1 = await data_loader.split_data(
            sample_dataframe, target_col='price'
        )
        X_train2, X_test2, y_train2, y_test2 = await data_loader.split_data(
            sample_dataframe, target_col='price'
        )
        
        # Same random_state should produce identical splits
        assert X_train1.equals(X_train2)
        assert X_test1.equals(X_test2)
        assert y_train1.equals(y_train2)
        assert y_test1.equals(y_test2)

    @pytest.mark.asyncio
    async def test_split_data_with_different_random_state(self, data_loader, sample_dataframe):
        """Test that different random_state produces different splits."""
        X_train1, _, y_train1, _ = await data_loader.split_data(
            sample_dataframe, target_col='price'
        )
        
        # Create new loader with different random_state
        data_loader2 = DataLoader(data_loader.data_path, test_size=0.33, random_state=99)
        X_train2, _, y_train2, _ = await data_loader2.split_data(
            sample_dataframe, target_col='price'
        )
        
        # Different random_state should produce different splits
        assert not X_train1.equals(X_train2)
        assert not y_train1.equals(y_train2)