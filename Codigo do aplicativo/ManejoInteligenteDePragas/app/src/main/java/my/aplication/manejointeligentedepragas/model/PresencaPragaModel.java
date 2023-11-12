package my.aplication.manejointeligentedepragas.model;

public class PresencaPragaModel {
    private int Cod_PresencaPraga;
    private int Status;
    private int Sync_Status;
    private int fk_Cod_Praga;
    private int fk_Cod_Talhao;

    public PresencaPragaModel(int fk_Cod_Praga, int fk_Cod_Talhao, int cod_PresencaPraga, int status, int sync_Status) {
        this.fk_Cod_Praga = fk_Cod_Praga;
        this.fk_Cod_Talhao = fk_Cod_Talhao;
        this.Cod_PresencaPraga = cod_PresencaPraga;
        this.Status = status;
        this.Sync_Status = sync_Status;
    }

    public PresencaPragaModel() {
        this.fk_Cod_Praga = 0;
        this.fk_Cod_Talhao = 0;
        this.Cod_PresencaPraga = 0;
        this.Status = 0;
        this.Sync_Status = 0;
    }

    public int getFk_Cod_Praga() {
        return fk_Cod_Praga;
    }

    public void setFk_Cod_Praga(int fk_Cod_Praga) {
        this.fk_Cod_Praga = fk_Cod_Praga;
    }

    public int getFk_Cod_Talhao() {
        return fk_Cod_Talhao;
    }

    public void setFk_Cod_Talhao(int fk_Cod_Talhao) {
        this.fk_Cod_Talhao = fk_Cod_Talhao;
    }

    public int getCod_PresencaPraga() {
        return Cod_PresencaPraga;
    }

    public void setCod_PresencaPraga(int cod_PresencaPraga) {
        Cod_PresencaPraga = cod_PresencaPraga;
    }

    public int getStatus() {
        return Status;
    }

    public void setStatus(int status) {
        Status = status;
    }

    public int getSync_Status() {
        return Sync_Status;
    }

    public void setSync_Status(int sync_Status) {
        Sync_Status = sync_Status;
    }
}
