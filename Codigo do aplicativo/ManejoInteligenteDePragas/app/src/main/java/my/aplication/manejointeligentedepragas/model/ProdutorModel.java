package my.aplication.manejointeligentedepragas.model;

public class ProdutorModel {
    private int Cod_Produtor;
    private int fk_Cod_usuario;

    public ProdutorModel(int cod_Produtor, int fk_Cod_usuario) {
        this.Cod_Produtor = cod_Produtor;
        this.fk_Cod_usuario = fk_Cod_usuario;
    }

    public int getCod_Produtor() {
        return Cod_Produtor;
    }

    public void setCod_Produtor(int cod_Produtor) {
        this.Cod_Produtor = cod_Produtor;
    }

    public int getFk_Cod_usuario() {
        return fk_Cod_usuario;
    }

    public void setFk_Cod_usuario(int fk_Cod_usuario) {
        this.fk_Cod_usuario = fk_Cod_usuario;
    }
}
